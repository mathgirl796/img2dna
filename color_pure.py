from PIL import Image

rows = 256 + 4
cols = 256 + 2
# 创建一个256x256的图像，背景色为群青色
image = Image.new('RGB', (cols, rows), (102, 153, 204))

# 保存图像
image.save('color_pure.png')