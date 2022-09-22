import re
import os

res_txt=r"D:\anaconda3_python37\Github\yolov5-5.0\try"
def debug_arr_tran(data_txt,res_path=res_txt):
    # 设置(     小数    )的匹配模式
    txt=os.path.split(data_txt)[1] # 获取带有文件扩展名的txt
    res_txt=os.path.join(res_path,txt)

    pattern="(\d+.?\d*)"
    with open(data_txt,"r") as f:
        # line_num=len(f.readlines())
        for index,line in enumerate(f.readlines()):
            line_res=re.compile(pattern).findall(line)
            print(line_res)

if __name__=="__main__":
    data_txt=r"D:\anaconda3_python37\Github\yolov5-5.0\nms_data"
    debug_arr_tran(data_txt)

