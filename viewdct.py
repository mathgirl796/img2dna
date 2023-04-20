import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# 从命令行读取图像
img_path = sys.argv[1]
img = cv2.imread(img_path)

# 将图像转换为YUV格式
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 分离通道，清理格式
y, u, v = cv2.split(yuv_img)
y, u, v = np.float32(y) - 128, np.float32(u) - 128, np.float32(v) - 128

# 对每个通道进行DCT变换

def dct88(src):
    dst = np.ones_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            dst[i:i+8, j:j+8] = cv2.dct(src[i:i+8, j:j+8])
    return dst

y_dct = dct88(y)
u_dct = dct88(u)
v_dct = dct88(v)

# 将DCT系数格式化到0-255之间
y_dct_norm = cv2.normalize(y_dct, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
u_dct_norm = cv2.normalize(u_dct, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
v_dct_norm = cv2.normalize(v_dct, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 填充空行列，为了显示起来好看
def expand_array(arr):
    # 获取数组的形状
    rows, cols = arr.shape

    # 计算插入空行和空列的数量
    num_rows_to_insert = (rows // 8) + 1
    num_cols_to_insert = (cols // 8) + 1
    
    # 插入空行
    for i in range(num_rows_to_insert):
        row_to_insert = np.full((1, 1), 128, dtype=arr.dtype)
        arr = np.insert(arr, (i*8)+i, row_to_insert, axis=0)
    
    # 插入空列
    for i in range(num_cols_to_insert):
        col_to_insert = np.full((1, 1), 128, dtype=arr.dtype)
        arr = np.insert(arr, (i*8)+i, col_to_insert, axis=1)

    return arr

# 显示灰度图像
expand_dct_norm = np.hstack((expand_array(y_dct_norm),expand_array(u_dct_norm),expand_array(v_dct_norm)))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1600, 900)
cv2.imshow('image', expand_dct_norm)

# 等待按下任意按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()