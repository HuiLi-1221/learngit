# 1. 读取并打开一张图片
# 2. 将图片转换为tensor, 并显示其形状
# 3. 调整图片的大小, 并显示其形状
# 4. 保存图片
#
# 5. 搭建一个有两个卷积层和一个激活层的网络
# 6. 加载图片, 并将其输入到网络中, 并显示输出的形状
#
# 呆呆宝加油!

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. 读取并打开一张图片
image_path = "1.png"
img = Image.open(image_path)
#img.show()

# 2. 将图片转换为tensor，并显示其形状
transform = transforms.ToTensor()
img_tensor = transform(img)
print("Original image tensor shape:", img_tensor.shape)

# 3. 调整图片的大小，并显示其形状
resize_transform = transforms.Resize((2048, 2048))
resized_img = resize_transform(img)
resized_tensor = transform(resized_img)
print("Resized image tensor shape:", resized_tensor.shape)

# 4. 保存图片
resized_img.save("resized_image.jpg")
#img.show()

# 5. 搭建一个有两个卷积层和一个激活层的网络
class HuibaoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=5, padding=3)  # [16, 205, 205]
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=3, padding=1)  # [32, 68, 68]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# 6. 加载图片, 并将其输入到网络中, 并显示输出的形状
model = HuibaoNet()
img_tensor = transform(img).unsqueeze(0)
print(img_tensor.shape)
output_tensor = model(img_tensor)
print(output_tensor.shape)


if __name__ == "__main__":
    print(torch.__version__)
