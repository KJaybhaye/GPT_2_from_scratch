import os
# import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

local_dir = "fineweb_edu"
shard_size = int(1e8)
# shard_size = int(1e4)
remote_name = "sample-10BT"

DATA_CACHE_DIR = os.path.join("./", local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
eot = tokenizer.encode(tokenizer.eos_token)

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot[0]] # the special <|endoftext|> token delimits all documents

    tokens.extend(list(tokenizer.encode(doc["text"])))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


# nprocs = max(1, os.cpu_count()//2)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

# for tokens in pool.map(tokenize, fw, chunksize=16):
for doc in fw:
    # if shard_index >= 11:
    #     break
    tokens = tokenize(doc)
    if token_count + len(tokens) < shard_size:
        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # write the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

        np.save(filename, all_tokens_np)
        
        shard_index += 1
        progress_bar = None
        # populate the next shard with the leftovers of the current doc
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder