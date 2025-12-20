import torch
import torch.nn as nn
import torch.nn.functional as F


BLOCK_SIZE = 512
BATCH_SIZE = 16
EMBED_SIZE = 256
HEADS = 4
LAYERS = 4
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"