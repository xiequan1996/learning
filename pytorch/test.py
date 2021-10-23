'''
Descripttion: 
version: 
Author: xiequan
Date: 2021-10-22 15:37:32
LastEditors: Please set LastEditors
LastEditTime: 2021-10-22 21:14:54
'''
import torch
import numpy as np
from torch import nn

x = np.arange(150)
y = x.reshape(2, 3, 5, 5)
z = torch.tensor(y, dtype=torch.float32)
z1 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=0)(z)
z2 = nn.Conv2d(3, 2, kernel_size=2, stride=2, padding=0)(z)
print(z1)
print(z2)
