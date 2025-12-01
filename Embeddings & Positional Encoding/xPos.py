import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class XPosRoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super().__init__()
        self.d_model = d_model
        
        # 1. Standard RoPE Frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # 2. Decay Calculation (Range 0.90 to 0.999)
        # This ensures distant tokens have less impact/importance
        min_decay = 0.90
        max_decay = 0.999
        indices = torch.arange(0, d_model, 2).float() / d_model
        gamma = min_decay + (max_decay - min_decay) * indices
        
        # Create scale: gamma ^ t
        scale = gamma.unsqueeze(0) ** t.unsqueeze(1)
        scale = torch.cat((scale, scale), dim=-1)
        
        # Register buffers
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.register_buffer("scale_cached", scale)

    def forward(self, x):
        seq_len = x.shape[1]
        
        # Slicing
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        scale = self.scale_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        # Rotation
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply Rotation AND Scaling (Decay) simultaneously
        return ((x * cos) + (x_rotated * sin)) * scale


if __name__ == "__main__":
    d_model = 64
    seq_len = 100
    heads = 4
    
    q = torch.ones(1, seq_len, heads, d_model)
    k = torch.ones(1, seq_len, heads, d_model)

    # Apply xPos
    xpos = XPosRoPE(d_model, max_seq_len=seq_len)
    q_x, k_x = xpos(q), xpos(k)

    score_xpos = torch.matmul(q_x[0, :, 0, :], k_x[0, :, 0, :].T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(score_xpos.detach().numpy(), cmap="mako")
    plt.title("xPos (RoPE + Decay)\n(Dampening effect on distant tokens)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()