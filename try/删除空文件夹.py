"""删除指定文件夹下的所有空文件夹"""
import os

spec_path = "../runs/detect"


def delete_empty_file():
    for root, dirs, files in os.walk(spec_path):
        if len(dirs) == 0 and len(files) == 0:
            os.rmdir(root)  # 将一层目录下


def total_delete_empty_file():
    delete_empty_file() # 删除第一级
    delete_empty_file()


total_delete_empty_file()
