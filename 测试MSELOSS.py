import torch.nn as nn
import torch
loss=nn.MSELoss(reduction="sum")
x=torch.tensor([1,2,3],dtype=torch.float32)
y=torch.tensor([2,3,4],dtype=torch.float32)
print(loss(y,x))