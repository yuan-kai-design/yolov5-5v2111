import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layers=nn.ModuleList([
            nn.Linear(1,10),
            nn.ReLU(),
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,1)
        ])

    def forward(self,x):
        out=x
        for layer in self.layers:
            out=layer(out)
        return out

if __name__=="__main__":
    x=torch.Tensor(1,2).reshape(2,-1).reshape(-1,1)
    x[0]=1;x[1]=2
    print(x)
    # m=model()
    # print(m(x))