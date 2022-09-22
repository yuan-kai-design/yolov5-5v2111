from pytesseract import image_to_string
from PIL import Image
a=image_to_string(image=Image.open(r"src/核酸检测.jpg"),lang="chi_sim")
print(a)