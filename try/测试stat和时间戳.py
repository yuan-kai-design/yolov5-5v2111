from pathlib import Path
import datetime
exp13_path=Path("../runs/detect/exp13")
timestamp=datetime.datetime.fromtimestamp(exp13_path.stat().st_mtime)
print(f"{timestamp.year}-{timestamp.month}-{timestamp.day}")
print(exp13_path.resolve()) # 返回绝对路径
# datetime.datetime.fromtimestamp()