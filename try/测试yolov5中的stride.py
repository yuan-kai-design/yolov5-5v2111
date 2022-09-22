
import torch
pt=torch.load(r"../params_pt/yolov5s.pt",map_location="cuda:0")
print("pt.stride:",pt["model"].stride)
print("pt.stride.max():",pt["model"].stride.max())
print(hasattr(pt["model"],"ch"))


