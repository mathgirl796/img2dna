# import numpy as np

# a = np.array([[1,1,1], 
#               [2,2,2], 
#               [3,3,3]])

# b = np.array([[50,50,50], 
#               [60,60,60], 
#                         ])

# c = np.array([[50,50   ], 
#               [60,60   ], 
#               [70,70   ]])

# d = np.array([[50,50   ], 
#               [60,60   ], 
#                         ])

# print(np.int8 == np.uint8)

# from itertools import permutations, product, combinations, combinations_with_replacement
# for x in product("ACGT", "ACGT", "ACGT", "ACGT"):
#     print(x)

# for x in combinations("ACTG", 2):
#     print(x)

# # for x in combinations_with_replacement("ACTG", 2):
# #     print(x)

# print(np.diagonal(a, -1).shape[0])

# a = np.array([3], dtype=np.float32)
# b = np.array([2], dtype=np.uint8)
# a[0] = b[0] * 999
# print(a)

from utils import *
# a = count_kmers("AAAATTTT", 4, canonical=True)
# print(a)

# import cv2
# import numpy as np

# # 读取彩色图像
# # img = cv2.imread('demo_data/color_pure.png')
# # img = cv2.imread('demo_data/lena_std.tif')
# img = cv2.imread('demo_data/len_std.jpg')
# original_shape = img.shape[:-1]

# # 进行转换
# c = bgr2yuv(img, addition=-128)
# channels = []
# for i, x in enumerate(cv2.split(c)):
#     x = pad(x, base=8)
#     pad_shape = x.shape
#     x = dct88(x)
#     x = quantify88(x, luminance_quantization_table if i == 0 else chrominance_quantization_table)
#     x = x.round()
#     x = dc_delta88(x)
#     x = zigzag88(x)

#     # 编码前先把数据编码为无符号整数
#     x = x.round().astype(np.uint8)
#     x = bytes2dna(x)
#     x = dna2bytes(x)

#     # 解码前先把数据变成有符号整数
#     x = x.astype(np.int8)
#     x = izigzag88(x, *pad_shape)
#     x = dc_idelta88(x)
#     x = iquantify88(x, luminance_quantization_table if i == 0 else chrominance_quantization_table)
#     x = idct88(x)
#     x = ipad(x, *original_shape)
#     channels.append(x)
# c = np.dstack(channels)
# c = yuv2bgr(c, addition=+128)

# # 打印转换后图像
# print(img[0][0], img.shape, img.dtype)
# print(c[0][0], c.shape, c.dtype)
# err = np.abs(img.astype(np.float32) - c.astype(np.float32))
# print(np.sum(err)/err.size, err.shape, np.sum(err > 100))
# cv2.imshow("origin", img)
# cv2.imshow("final", c)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# from collections import Counter
# c1 = Counter("AAAABB")
# c2 = Counter("AABBBB")
# c3 = c1 & c2
# print(c1 & c2)
# print(sum(c3.values()))

# filepath = "/home/dr/Desktop/img2DNA/p.1071/log_2023-04-22-02-19-55.txt"
# with open(filepath, "r", encoding="utf8") as f:
#     lines = f.readlines()
#     l = []
#     for line in lines:
#         if line.startswith("unique kmer: "):
#             l.append(int(line.strip().split()[2]))
# print(sum(l), len(l), sum(l)/len(l))

a = np.array([0,1,2, 3,4,5, 6,7,8, 9,10,11, 12,13,14])
print(a.reshape(-1, 3).T.reshape(-1))