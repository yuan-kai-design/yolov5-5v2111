from thop import profile
from torchvision.models import resnet50
from copy import deepcopy
import torch
model=resnet50()
print(model.modules())
flops=profile(deepcopy(model),inputs=(torch.zeros((1,3,224,224)),),verbose=False)
print(flops)