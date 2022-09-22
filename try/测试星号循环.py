import torch
a=torch.tensor(data=[[1,3,5,6,5,2]],dtype=torch.float)
print(a)
print(a.shape)
print("判断是几维张量：",len(a.shape))
for *i,j,k in a:
    print(*i,j,k)
    # [tensor(1.), tensor(3.), tensor(5.), tensor(6.)]
    # tensor(5.)
    # tensor(2.)

# 下面1个例子便于理解
for i,j,k in [(1,2,3),(4,5,6)]:
    print(i,j,k)
# 1 2 3
# 4 5 6

