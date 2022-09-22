import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(message)s--%(funcName)s"
)
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(os.environ)