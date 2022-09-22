import torch
a=torch.tensor([[ 1,  5, 62, 54],
        [ 2,  6,  2,  6],
        [ 2, 65,  2,  6]])
maxv,ind=torch.max(a,dim=1,keepdim=True)
print("maxv=",maxv)
print("ind=",ind)
maxv,ind=torch.max(a,dim=1)
print("maxv=",maxv)
print("ind=",ind)