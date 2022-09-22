import torch
y=torch.tensor([3,2,4,5,6,7,3,1,3,5,7,8,9,54,1,1,1,0,1])
a=torch.unique(y)
print(a) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 54]) 去除重复项
a=torch.unique(y,sorted=False)
print(a) # tensor([ 3,  2,  4,  5,  6,  7,  1,  8,  9, 54,  0])

z = torch.tensor([1,3,5,7,9,0,8,6,4,2,34,5,6,7,8,9,1,3,6,7,8,9,53,2,3])
b=torch.unique(z)
print(b) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 34, 53])
b=torch.unique(z,sorted=False)
print(b) # tensor([ 9,  1,  3,  5,  7,  8,  0,  6,  4,  2, 34, 53])