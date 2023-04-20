import cv2
from utils import *

import warnings
import sys

def warn_and_exit(message, category, filename, lineno, file=None, line=None):
    sys.exit(message)

warnings.simplefilter('always', RuntimeWarning)
warnings.showwarning = warn_and_exit
warnings.filterwarnings('error', category=RuntimeWarning)

"""
note:
Y[0-255]: dark_green-light_green
U[0-255]: green-blue
V[0-255]: green-red
img.shape: height, width, channels
yuv should be converted to np.float32 before thrown into dct()
img should be converted to np.uint8 before thrown into cv2.imshow()
"""

def img2dna(img:np.ndarray):
    """
    input: cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    output: str of dna sequence
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv = yuv
    img_shape = yuv.shape[:-1]
    jpeg_raws = []
    # get jpeg zigzag raw
    for i, c in enumerate(split_channels(yuv)):
        c = pad(c, base=8).astype(np.float32) # smoothly pad
        pad_shape = c.shape
        c = dct88(c)
        c = quantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table, np.int8)
        c = dc_delta88(c)
        c = zigzag88(c)
        jpeg_raws.append(c)
    
    # convert to dna
    dna = ""
    for raw in jpeg_raws:
        dna += bytes2dna(raw)
    
    # return
    return dna, img_shape, pad_shape

def dna2img(dna:str, img_shape, pad_shape):
    """
    img_shape, pad_shape: (height, width)
    """
    jpeg_raws = dna2bytes(dna).reshape(3, -1)
    channels = []
    for i, raw in enumerate(jpeg_raws):
        c = izigzag88(raw, *pad_shape)
        c = dc_idelta88(c)
        c = iquantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table, iquantify_type=np.float32)
        c = idct88(c).astype(np.uint8)
        c = ipad(c, *img_shape)
        channels.append(c)
    yuv = np.dstack(channels)
    print(yuv.shape, yuv.dtype)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    

img=cv2.imread("lena_std.tif", cv2.IMREAD_COLOR)
dna, img_shape, pad_shape = img2dna(img)
img = dna2img(dna, img_shape, pad_shape)
cv2.imshow("", img.astype(np.uint8))
cv2.waitKey(0) 
cv2.destroyAllWindows()

# img=cv2.imread("lena_std.tif", cv2.IMREAD_COLOR)
# img=cv2.imread("len_std.jpg", cv2.IMREAD_COLOR)
# print(f"img: {type(img)} {img.shape}")
# yuv=cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
# print(f"yuv: {type(yuv)} {yuv.shape}")

# print(len(img2dna(img)))

# b,g,r=split_channels(img)
# y,u,v=split_channels(yuv)

# shape = y.shape

# yp = pad(y)
# print("yp", yp.shape, yp)

# ypd = dct88(yp)
# print("ypd", ypd.shape, ypd)

# ypdq = quantify88(ypd, np.full((8,8), 10 ))
# print("ypdq", ypdq.shape, ypdq)

# ypdq_delta = dc_delta88(ypdq)
# print("ypdq_delta", ypdq_delta.shape, ypdq_delta, np.sum(ypdq_delta == 115))

# ypdqz = zigzag88(ypdq_delta)
# print("ypdqz", ypdqz.shape, ypdqz, np.sum(ypdq_delta == 115))

# dna = bytes2dna(ypdqz)
# print(dna, len(dna))

# cv2.imshow("", yp.astype(np.uint8))
# cv2.waitKey(0) 
# cv2.destroyAllWindows()