"""编一个小网络拟合正弦函数"""
import torch.nn as nn
from numpy import pi
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

x=torch.linspace(-2*pi,2*pi,400) # 400等分
input=x.view(400,-1).float()  # 使用resize时报错
output=torch.sin(input).float()

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layers=nn.ModuleList([
            nn.Linear(1,10),nn.SiLU(),
            nn.Linear(10,100),nn.SiLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        ])

    def forward(self,x):
        out=x
        for layer in self.layers:
            out=layer(out)
        return out

model1=model()
optim1=torch.optim.Adam(model1.parameters(),lr=0.05) # 定义优化器
Loss=nn.MSELoss() # 定义损失函数

for i in range(3000):
    pred=model1(input) # pred只是预测结果，参数存在model1中
    loss=Loss(pred,output)
    optim1.zero_grad()
    loss.backward()
    optim1.step()
    # 打印训练信息
    if (i+1)%100==0:
        print("training step:step {},loss {}".format(i+1,loss))
model1.eval()
torch.save(model1.state_dict(),r"../params_pt/fine_sin.pt") # 保存参数

# with torch.no_grad():
#     # 查看梯度下降3000次后的sin拟合情况
#     pred=model1(input)
#     print(model1.state_dict())
# plt.plot(x.numpy(), pred.numpy(), "r-")
# plt.plot(x.numpy(), output.numpy(), "g-")
# plt.show()



