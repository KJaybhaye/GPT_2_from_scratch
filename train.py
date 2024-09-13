import math
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2Tokenizer
import os
import time
from utils import *
from gpt import *
from dataloader import FinewebDataloader

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "cuda not available"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of processes / devices
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

print(f"ddp: {ddp}")
device_type = "cuda" if device.startswith("cuda") else "cpu"


total_batch_size = 64 * 1024 # in number of tokens
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# data_dir = "fineweb_edu"

train_dir = "/kaggle/input/fineweb-edu-10bt-for-gpt2/train"
val_dir = "/kaggle/input/fineweb-edu-10bt-for-gpt2/test"

# train_loader = FinewebDataloader(tokenizer, B, T, data_root = data_dir, process_rank=ddp_rank, 
#                                  num_processes=ddp_world_size, split='train', master_process=master_process)
# val_loader = FinewebDataloader(tokenizer, B, T, data_root = data_dir, process_rank=ddp_rank, 
#                                num_processes=ddp_world_size, split='val', master_process=master_process)

train_loader = FinewebDataloader(tokenizer, B, T, data_root = train_dir, process_rank=ddp_rank, 
                                 num_processes=ddp_world_size, separate_val=True, split='train', master_process=master_process)
val_loader = FinewebDataloader(tokenizer, B, T, data_root = val_dir, process_rank=ddp_rank, 
                               num_processes=ddp_world_size, separate_val=True, split='val', master_process=master_process)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100 #715
max_steps = train_loader.num_batches // (grad_accum_steps * ddp_world_size )
use_compile = False

if master_process:
    print(f"total steps: {max_steps}")
torch.cuda.empty_cache()
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = get_optimizer(model, max_lr, 0.1)

def dist_train_step(model, optimizer, train_loader, grad_accum_steps, device, step, log_file):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = next(train_loader)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # syncronise after last micro step only
            
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #     logits, loss = model(x.to(device), y.to(device))
        logits, loss = model(x.to(device), y.to(device))
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.batch_size * train_loader.seq_length * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

def dist_eval_step(model, optimizer, val_loader, device, step, log_file, log_dir = "./log", last_step = False):
    model.eval()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = next(val_loader)
            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            #     logits, loss = model(x.to(device), y.to(device))
            logits, loss = model(x.to(device), y.to(device))
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        
    if master_process:
        print(f"validation loss (step: {step}): {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
        if step > 0 and (step % 250 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            if ddp:
                st_dict = model.module.state_dict()
                config = model.module.config
            else:
                st_dict = model.state_dict()
                config = model.config
            checkpoint = {
                'model': st_dict,
                'config': config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)

def get_lr(it): # cosine schedular
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  if it > max_steps:
    return min_lr
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)


log_dir = "fineweb_log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

train_iter = iter(train_loader)
val_iter = iter(val_loader)

for step in range(max_steps):
    last_step = (step == max_steps - 1)
    if step % 100 == 0 or last_step:
        dist_eval_step(model, optimizer, val_iter, device, step, log_file, log_dir, last_step)

    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) and master_process:
        samples = sample("Tell me about nuclear energy,", model, tokenizer, num_return_sequences=1, max_length=50)
        for sam in samples:
              print(sam)

    dist_train_step(model, optimizer, train_iter, grad_accum_steps, device, step, log_file)


if ddp:
    destroy_process_group()
