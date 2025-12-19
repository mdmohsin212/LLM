import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random
from pathlib import Path
import json


HEADS = 4
LAYERS = 4
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_state(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_ids(ids, pair, idx):
    newids=[]
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    
    return newids

def train_bpe(text, vocab_size):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    ids = list(tokens)
    num_merges = vocab_size - 256
    merges = {}
    for i in range(num_merges):
        stats = get_state(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge_ids(ids, pair, idx)
        merges[pair] = idx
    return merges

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(text) >= 2:
        stats = get_state(ids)
        pair = min(stats, key=lambda p:merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        ids = merge_ids(ids, pair, idx)
    return ids

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")

if Path('input.txt').exists():
    text = Path('input.txt').read_text("utf-8")

merges = train_bpe(text, VOCAB_SIZE)

vocab_data = {f"{p[0]},{p[1]}": idx for p, idx in merges.items()}
with open("vocab.json", "w") as f:
    json.dump(vocab_data, f)
    

data_ids = encode(text, merges)
data = torch.tensor(data_ids, dtype=torch.long)


def get_batch():
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure embed size is divisible by number of heads
        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)

        # Output projection after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Project Q, K, V
        queries = self.W_q(query)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Split into heads: (N, heads, seq_len, head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.reshape(N, value_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1))
        energy = energy / math.sqrt(self.head_dim)

        # Optional mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention weights
        attention = torch.softmax(energy, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attention, values)

        # Merge heads: (N, seq_len, embed_size)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final output projection
        out = self.fc_out(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1, forward_expansion=4):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = RMSNorm(embed_size)
        self.norm2 = RMSNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, embed_size * forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention_input = self.norm1(query)
        attention_output = self.attention(attention_input, attention_input, attention_input, mask)
        x = query + self.dropout(attention_output)
        ffn_input = self.norm2(x)
        fnn_output = self.feed_forward(ffn_input)
        output = x + self.dropout(fnn_output)
        return output
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, layers, heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, embed_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, heads) for _ in range(layers)
        ])
        
        self.ln_f = RMSNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        mask = torch.tril(torch.ones((T, T), device=idx.device)).expand(
            B, 1, T, T
        )
        
        for block in self.blocks:
            x = block(x, x, x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = TransformerLM(VOCAB_SIZE, EMBED_SIZE, LAYERS, HEADS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

for step in range(MAX_ITERS):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")