import torch
import numpy as np
from torch import nn

x = np.arange(150)
y = x.reshape((2, 3, 5, 5))
z = torch.tensor(y, dtype=torch.float32)
z1 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=0)(z)
z2 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=0)(z)
print(z1)
print(z2)
