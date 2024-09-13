import torch
import numpy as np
import os

class DistributedDataloader: # for text file
  def __init__(self, txt, tokenizer, batch_size, seq_length, process_rank = 0 , num_processes = 1):
    with open(txt, 'r', encoding='utf-8') as f:
      text = f.read()
    self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    self.n_tokens = len(self.tokens)
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.num_batches = self.n_tokens // (batch_size * seq_length)

    self.ind = batch_size * seq_length * process_rank
    
    self.process_rank = process_rank
    self.num_processes = num_processes

    print("Number of tokens: ", self.n_tokens)
    print("Number of Batches: ", self.num_batches)

  def __iter__(self):
    return self

  def __len__(self):
    return self.num_batches

  def __next__(self):
    B, T = self.batch_size, self.seq_length
    if self.ind >= self.n_tokens - B * T * self.num_processes - 1:
      self.ind = self.batch_size * self.seq_length * self.process_rank
#       raise StopIteration
    BT = self.tokens[self.ind : self.ind + B * T + 1]
    self.ind += B * T * self.num_processes

    x = BT[:-1].view(B, T)
    y = BT[1:].view(B, T)

    return x, y

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class FinewebDataloader:
  def __init__(self, tokenizer, batch_size, seq_length, data_root = "edu_fineweb10B", process_rank = 0, 
               num_processes = 1, separate_val = False, split='train', master_process=True):
    self.batch_size = batch_size
    self.seq_length = seq_length

    self.ind = batch_size * seq_length * process_rank

    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}

    shards = os.listdir(data_root)
    
    if not separate_val:
        shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]

    self.shards = shards
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    assert len(shards) > 0, f"no shards found for split {split}"
    if master_process:
        print(f"found {len(shards)} shards for split {split}")
    # self.reset()

    data = [np.load(f, mmap_mode='r') for f in shards]
    self.n_tokens = sum([d.shape[0] for d in data])
    self.num_batches = self.n_tokens // (batch_size * seq_length)

    print("Number of tokens: ", self.n_tokens)
    print("Number of Batches: ", self.num_batches)

  def __iter__(self):
    return self

  def __len__(self):
    return self.num_batches

  def __next__(self):
    B, T = self.batch_size, self.seq_length

    if self.ind >= len(self.tokens)- B * T * self.num_processes - 1:
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.ind = self.batch_size * self.seq_length * self.process_rank
      # self.ind = 0
#       raise StopIteration
    BT = self.tokens[self.ind : self.ind + B * T + 1]
    self.ind += B * T * self.num_processes

    x = BT[:-1].view(B, T)
    y = BT[1:].view(B, T)

    return x, y