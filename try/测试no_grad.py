import torch
a=torch.tensor([1.1],requires_grad=True)
print(a)
a.add_(2)
b=a*2
print(b)
