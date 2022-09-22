import torch
a1=torch.zeros(1,3,640,640)
a1.type(torch.FloatTensor)
# a1=a1.type(torch.DoubleTensor)
print(a1.data.type())
a3=torch.DoubleTensor(1,3,640,640)
print(a3.data.type())
print("将a1转变为与a3相同类型：",a1.to(torch.device("cuda:0")).type_as(a3).data.type())
print(a1.data.type())
