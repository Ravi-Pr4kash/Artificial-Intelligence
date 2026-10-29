import numpy as np
import torch
import torch.nn as nn
import time
import math


sentence = 4
token = 10
dimension = 512
heads = 8
head_dim = dimension // heads

x = torch.randn(sentence, token, dimension)
print(x.shape)

x_heads = x.view(sentence, token, heads, head_dim)
print(x_heads.shape)

x_ready = x_heads.permute(0, 2, 1, 3)
print(f"After permute for attention: {x_ready.shape}")