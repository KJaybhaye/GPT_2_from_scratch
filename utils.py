import inspect
import torch
import torch.nn.functional as F

def sample(input_string, model, tokenizer, device, num_return_sequences=1, max_length=200):
  tokens = tokenizer.encode(input_string)
  tokens = torch.tensor(tokens, dtype=torch.long)
  tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

  x = tokens.to(device)
  while x.size(1) < max_length:
    with torch.no_grad():
      logits, _ = model(x)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)

      topk_probs, topk_ix = probs.topk(50)
      ix = torch.multinomial(topk_probs, num_samples=1)
      xcol = torch.gather(topk_ix, dim=-1, index=ix)
      x = torch.cat((x, xcol), dim=1)

  outputs = []
  for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    outputs.append(tokenizer.decode(tokens))

  return outputs


def get_optimizer(model, lr, weight_decay=0.1, device='cpu', betas = (0.9, 0.95)):
  params_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
  # all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
  decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
  nodecay_params = [p for n, p in params_dict.items() if p.dim() < 2]

  optim_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": nodecay_params, "weight_decay": 0.0},
  ]
  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and device == 'cuda'

  optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=use_fused)
  return optimizer



