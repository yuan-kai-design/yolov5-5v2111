import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        for i in range(4):
            setattr(self,"ReLU{}".format(i),nn.SiLU(inplace=True))


    def forward(self,x):
        for i in range(4):
            x=getattr(self,)

model1=model()
print(list(model1.modules()))
