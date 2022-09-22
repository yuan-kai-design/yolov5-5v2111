from torch import squeeze,unsqueeze,arange
a=arange(1,17,1).resize(4,2,2) # (C,H,W)
print(a.shape)
b=unsqueeze(a,1)
b=unsqueeze(b,1)
# print(b)
print(b.shape)
a1=squeeze(b,1)
print(a1.shape)