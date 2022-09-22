import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
dd=torch.device("1")
dd_type=dd.type
print(dd_type)