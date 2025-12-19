import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, embd_size, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embd_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embd_size)
        self.dropout = nn.Dropout(dropout)
        



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
        
        # self.norm1 = nn.LayerNorm(embed_size)
        # self.norm2 = nn.LayerNorm(embed_size)
        
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
    

if __name__ == "__main__":
    embed_size = 512
    heads = 8
    dropout = 0.1
    
    x = torch.randn(1, 10, 512)
    
    block = TransformerBlock(embed_size, heads, dropout)
    
    output = block(x, x, x, mask=None)
    
    print("Transformer Block Test:")
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")