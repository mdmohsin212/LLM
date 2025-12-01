import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns

def get_alibi_bias(num_heads, seq_len):
    # Create distance matrix: |i - j|
    context_pos = torch.arange(seq_len)[:, None]
    memory_pos = torch.arange(seq_len)[None, :]
    relative_pos = memory_pos - context_pos
    
    # Only consider negative values (past context) as penalty
    distance = torch.abs(relative_pos).float() * -1 

    # Generate slopes for heads: 1/2, 1/4, 1/8...
    slopes = torch.tensor([1.0 / (2 ** (i + 1)) for i in range(num_heads)])
    
    # Bias = Slope * Distance
    # Shape: (1, Heads, Seq, Seq)
    bias = slopes.view(1, num_heads, 1, 1) * distance.view(1, 1, seq_len, seq_len)
    return bias

if __name__ == "__main__":
    d_model = 64
    seq_len = 100
    heads = 4
    
    # Dummy Input
    q = torch.ones(1, seq_len, heads, d_model)
    k = torch.ones(1, seq_len, heads, d_model)

    # 1. Calculate raw dot product score
    raw_score = torch.matmul(q[0, :, 0, :], k[0, :, 0, :].T) / math.sqrt(d_model)
    
    # 2. Calculate ALiBi bias
    alibi_bias = get_alibi_bias(heads, seq_len)[0, 0] # Bias for Head 0
    
    # 3. Add bias to the raw score
    score_alibi = raw_score + alibi_bias

    plt.figure(figsize=(8, 6))
    sns.heatmap(score_alibi.detach().numpy(), cmap="magma")
    plt.title("ALiBi Attention\n(Linear penalty added directly to score)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()