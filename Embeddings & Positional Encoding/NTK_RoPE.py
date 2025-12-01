import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class NTKRoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000, alpha=8.0):
        super().__init__()
        # NTK Scaling: Modify base frequency to extend context length
        # alpha = scaling factor (e.g., 8x context)
        base = base * alpha ** (d_model / (d_model - 2))
        
        # Calculate frequency with new base
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # Cache sin/cos
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        # Slice cached values
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        # Rotation logic
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated * sin)


if __name__ == "__main__":
    d_model = 64
    seq_len = 100
    heads = 4
    
    q = torch.ones(1, seq_len, heads, d_model)
    k = torch.ones(1, seq_len, heads, d_model)

    # Apply NTK RoPE (Alpha = 8.0)
    ntk_rope = NTKRoPE(d_model, max_seq_len=seq_len, alpha=8.0)
    q_ntk, k_ntk = ntk_rope(q), ntk_rope(k)

    score_ntk = torch.matmul(q_ntk[0, :, 0, :], k_ntk[0, :, 0, :].T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(score_ntk.detach().numpy(), cmap="plasma")
    plt.title("NTK Scaled RoPE\n(Optimized for Long Context)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()