from pathlib import Path
import os
parent_file=Path(os.getcwd()).parent
print(type(parent_file)) # WindowsPath
print(parent_file)
