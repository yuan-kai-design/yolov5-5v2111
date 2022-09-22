import torch
a,b=torch.meshgrid(torch.tensor([1,2,3]),torch.tensor([4,5,6]))
print(a)
print(b)
print(torch.stack((torch.tensor([[0]]),torch.tensor([[0]])),dim=2))