import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super().__init__()
        # Calculate inverse frequency for RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        t = torch.arange(max_seq_len).float()
        # Outer product of time steps and frequencies
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate frequencies for sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register buffers to avoid repeated calculations
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Heads, Head_Dim)
        seq_len = x.shape[1]
        
        # Slice cached values based on current sequence length
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        # Apply rotation
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x):
        # Split vector into two parts and rotate (-x2, x1)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


if __name__ == "__main__":
    # Configuration
    d_model = 64
    seq_len = 100
    heads = 4
    
    # Dummy Input (All ones to isolate positional effect)
    q = torch.ones(1, seq_len, heads, d_model)
    k = torch.ones(1, seq_len, heads, d_model)

    # Apply RoPE
    rope = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
    q_r, k_r = rope(q), rope(k)

    # Calculate Attention Score (For Head 0 only)
    score_rope = torch.matmul(q_r[0, :, 0, :], k_r[0, :, 0, :].T)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(score_rope.detach().numpy(), cmap="viridis")
    plt.title("Standard RoPE Attention Score\n(Effect of Relative Distance)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()