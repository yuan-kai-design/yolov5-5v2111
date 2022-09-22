import torch
import yaml

x = torch.load("yolov5m.pt", map_location=torch.device("cpu"))
print(x)
# print(x.yaml)
# print(type(x))
# yaml_path = r"D:\anaconda3_python37\Github\yolov5-5.0\models\yolov5s.yaml"
# with open(yaml_path) as f:
#     yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
#     print(type(yaml_file)) # dict
#     for k in

