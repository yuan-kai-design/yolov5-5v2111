import torch
from pathlib import Path
from numpy import pi
import torch.nn as nn
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layers=nn.ModuleList([
            nn.Linear(1,10),nn.SiLU(),
            nn.Linear(10,100),nn.SiLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        ])

    def forward(self,x):
        out=x
        for layer in self.layers:
            out=layer(out)
        return out
import os
w=Path(r"D:\anaconda3_python37\Github\yolov5-5.0\params_pt\fine_sin_toal_model.pt")
# w_pt=os.path.join(w,"yolov5s.pt")
pt_info=torch.load(f=w) # map_location=torch.device("cpu")
print(pt_info)
w=Path(r"D:\anaconda3_python37\Github\yolov5-5.0\params_pt\yolov5l.pt")
pt_info=torch.load(f=w,map_location="cuda:0")
print(pt_info)
# for i in pt_info['model'].modules():
#     print(i)
