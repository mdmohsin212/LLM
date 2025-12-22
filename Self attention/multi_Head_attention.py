import torch
import torch.nn as nn
import math

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


if __name__ == "__main__":
    embed_size = 512
    heads = 8

    x = torch.randn((1, 10, 512))
    model = MultiHeadAttention(embed_size, heads)

    output = model(x, x, x, mask=None)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")