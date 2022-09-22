from torchvision.models import resnet101
import torch
import torch.nn as nn
from torchsummary import summary

model_dict=torch.load(r"../weights/resnet101.pt",map_location="cuda:0")
# print(model)
# print(model.fc.weight.shape)
# print(model.fc.bias.shape)
# print(model.fc.in_features)
# print(model.fc.out_features)
# summary(model,input_size=(3,224,224))
# print(model.parameters())
# print(model)
inp=torch.randn(1,3,224,224).cuda()
out=model(inp)
print(out.shape)
# print(nn.Parameter())