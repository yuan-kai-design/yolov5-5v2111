import torch
a=torch.tensor([1,2,3],dtype=torch.float64)
b=[a]+[a,a,a]
print(b)