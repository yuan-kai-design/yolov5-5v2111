import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layers=nn.ModuleList([
            nn.Linear(1,10),nn.SiLU(),
            nn.Linear(10,100),nn.SiLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        ])

    # def forward(self,x):
    #     out=x
    #     for layer in self.layers:
    #         out=layer(out)
    #     return out

# modeln=model()
# modeln.load_state_dict(torch.load(r"../params_pt/fine_sin.pt"))
# torch.save(modeln,r"../params_pt/fine_sin_toal_model.pt")
