from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
  vocab_size: int = 50257
  block_size: int = 1024 #max number of tokens in input and output
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768


class FF(nn.Module):
    def __init__(self, config: GPTConfig):
        super(FF, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # input and output neurons, inp to 4 inp
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPTSCALED_INIT = True # to compensate the increase of standard deviation in residual stream

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # n_embd = num_heads * head_size
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #key, value, query projection (3*k size)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd) #output projection
    self.c_proj.GPTSCALED_INIT = True

    self.n_head = config.n_head
    self.n_embd = config.n_embd

    #for parameters that shouldent be updated by optimizer but should be in state_dict (here mask)
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                         .view(1, 1, config.block_size, config.block_size))  # mask with size max_tokens x max_tokens

    #torch.tril Returns lower triangular part of the matrix or batch of matrices input, other elements are set to 0


  def forward(self, x):
    B, T, C = x.size() #shape is alise for size (batch, sequence length, n_embd)
    # print(f"b: {B}, t: {T}, c: {C}")
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # as out of c_att is all three concatnated

    # n_embd = num_heads * head_size
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # attn = (q @ k.transpose(-2, -1)) * (1.0/ math.sqrt(k.size(-1))) # (B, nh, T, T)
    # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #masking
    # attn = attn.softmax(dim=-1)
    # y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    return self.c_proj(y)


class Block(nn.Module): #decoder block
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = SelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)

    self.mlp = FF(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict({
        'wte': nn.Embedding(config.vocab_size, config.n_embd), # token embeddings vocab size x embedding dimension
        'wpe': nn.Embedding(config.block_size, config.n_embd),  # positional embeddings
        'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n_layer (6) decoder blocks
        'ln_f': nn.LayerNorm(config.n_embd),
        })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #final head embedding dims x vocab size

    # weight sharing scheme
    self.transformer['wte'].weight = self.lm_head.weight

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'GPTSCALED_INIT'):
        if module.GPTSCALED_INIT:
          std *= (2 * self.config.n_layer) ** -0.5
      nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x, targets=None):
    B, T = x.size()
    assert T <= self.config.block_size, "Embedding size is wrong"
    tok_emb = self.transformer['wte'](x)
    # print(tok_emb.shape)
    pos_emb = self.transformer['wpe'](torch.arange(0, T, device=x.device))
    # print(pos_emb.shape)
    x = tok_emb + pos_emb
    # print('tok + pos', x.shape)
    for block in self.transformer['h']:
      x = block(x)

    # print('blocks out', x.shape)
    x = self.transformer['ln_f'](x)
    # print('ln_f out', x.shape)
    logits = self.lm_head(x)
    # print('lm_head out', logits.shape)
    loss = None
    if targets is not None:
      loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss

  @classmethod
  def from_pretrained(cls, model_type):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
          'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
          'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
          'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
          'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
      }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
      # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])

    return model