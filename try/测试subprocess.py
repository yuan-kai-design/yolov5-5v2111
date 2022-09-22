import subprocess
from pathlib import Path
path=Path(__file__).parent
s= f"git -C {path} describe --tags --long --always"
obj=subprocess.check_output(s,shell=True,stderr=subprocess.STDOUT).decode()
print(obj)