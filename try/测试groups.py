import glob
from pathlib import Path
import re
path=Path(r"../runs/detect/exp").resolve() # 将相对地址转化为绝对地址
exp_path=glob.glob(rf"{path}*")
print(exp_path)
exp=[re.search(rf"exp(\d*)",m) for m in exp_path]
print(exp)
