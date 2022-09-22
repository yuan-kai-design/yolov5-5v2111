from pathlib import Path
import glob

exp_path=Path(r"D:\anaconda3_python37\Github\yolov5-5.0\runs\detect\exp")
print(exp_path.stem)
dir_path=glob.glob(rf"{exp_path}*")
print(dir_path)
