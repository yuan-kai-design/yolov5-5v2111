import cv2
import numpy as np
import matplotlib.pyplot as plt
# img_path=r"../data/images/office2.jpg"
# img=cv2.imread(filename=img_path)
# print(img.shape)
# 创建一个全黑图像

# zero_=np.zeros((20,10))
# zero_[2][3]=1
# plt.imshow(zero_)
# # plt.xticks([])
# # plt.yticks([])
# plt.axis("off")
# plt.savefig("zero_img.jpg",bbox_inches="tight")
# plt.show()
#
# # 读取
# img=plt.imread("zero_img.jpg")
# plt.axis("off")
# print(img.shape)
# plt.imshow(img) # 发现和原图的坐标不一样了，分辨率改变了
# plt.savefig("zero_img_not_axis.jpg",bbox_inches="tight")
# # plt.show()
#
#
img_new=cv2.imread(filename="zero_gray.jpg",flags=cv2.IMREAD_GRAYSCALE)
img_new[8][9]=180

cv2.circle(img_new,(9,8),radius=1,color=128,thickness=4)
print(img_new.shape)
# cv2.imwrite("zero_gray.jpg",img=img_new)
# cv2.imshow("img",img_new)
# cv2.waitKey()

img_new=cv2.resize(img_new,dsize=(20,10),interpolation=cv2.INTER_CUBIC)
print(img_new.shape)

cv2.imwrite("img_size.jpg",img_new)
# cv2.imshow("img",img_new)
#
# cv2.waitKey()


# print(img_new)
# print(img_new.shape)
# a_noteq255=np.where(img_new!=255)
# a_stack=np.column_stack(a_noteq255)
# print(a_stack.shape)
# print(a_stack)
# for coord in a_stack:
#     cv2.circle(img_new,(coord[1],coord[0]),radius=1,thickness=4,color=(128,255,255))
# cv2.imshow("img",img_new)
# cv2.waitKey(0)


