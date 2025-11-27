import torch
import torch.nn as nn
import math

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        # Shape: (Batch, Seq_Len, Embed_Size)
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)

        # Similarity Score -> Q * K^T
        attention_score = torch.bmm(Q, K.transpose(1, 2))
        # Shape: (Batch, Seq_Len, Seq_Len)

        # Scaling
        attention_score = attention_score / math.sqrt(self.embed_size)

        # Softmax (0-1)
        attention_weights = torch.softmax(attention_score, dim=-1)

        # Weights * V
        output = torch.bmm(attention_weights, V)
        return output

if __name__ == "__main__":
    batch_size = 1
    seq_len = 3
    embed_size = 4

    x = torch.rand((batch_size, seq_len, embed_size))

    single_head = SingleHeadSelfAttention(embed_size)
    output = single_head(x)

    print("Input : ", x)
    print("Input Shape : ", x.shape)
    print("Output Shape : ", output.shape)
    print("Final Output : ", output)