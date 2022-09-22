import numpy as np
import torch
import sys
sys.maxsize=10
torch.set_printoptions(precision=8,threshold=9)
a=torch.randn((2,3))
print(a)
np.set_printoptions(precision=8,threshold=200)
# b=np.random.randn((2,3))
# print(b)
c=np.arange(1,100)
print(c)
