import numpy as np
import cv2
from itertools import product
from collections import Counter
import time
import datetime
import sys
import os
import numpy as np

#########################
## image process
#########################
def pad(src:np.ndarray, base = 8):
    """
    input: 2d ndarray
    output: same dtype 2d ndarray. pad h,w to multiple of 8
    smoothly pad array h,w to multiple of base
    """
    x_pad = (base - src.shape[0]%base) % base
    y_pad = (base - src.shape[1]%base) % base
    ret = np.zeros((src.shape[0] + x_pad, src.shape[1] + y_pad), dtype=src.dtype)
    ret[0:src.shape[0], 0:src.shape[1]] = src
    for i in range(x_pad):
        ret[src.shape[0] + i, :] = ret[src.shape[0] - 1, :]
    for i in range(y_pad):
        ret[:, src.shape[1] + i] = ret[:, src.shape[1] - 1]
    return ret

def ipad(src:np.ndarray, original_height, original_width):
    """
    input: 2d ndarray
    output: same dtype 2d ndarray of src upper left original_shape pixels
    """
    return src[:original_height, :original_width]

def dct88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size np.float32 ndarray
    perform dct to each 8*8block (won't automatically pad)
    """
    src = src.astype(np.float32)
    dst = np.zeros_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            dst[i:i+8, j:j+8] = cv2.dct(src[i:i+8, j:j+8])
    return dst

def idct88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size np.float32 ndarray
    perform idct to each 8*8block (won't automatically pad)
    """
    src = src.astype(np.float32)
    dst = np.zeros_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            dst[i:i+8, j:j+8] = cv2.idct(src[i:i+8, j:j+8])
    return dst

luminance_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

chrominance_quantization_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def quantify88(src:np.ndarray, quantification_table:np.ndarray):
    """
    input: src - 2d ndarray. quantification_table: 8x8 ndarray
    output: 2d np.float32 ndarray. (won't automatically pad or round)
    perform src / q_table on each 8x8 block
    """
    assert quantification_table.shape == (8, 8)
    src = src.astype(np.float32)
    dst = np.zeros_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ir, jr = np.min((i+8, src.shape[0])), np.min((j+8, src.shape[1]))
            dst[i:i+8, j:j+8] = src[i:i+8, j:j+8] / quantification_table[:ir-i, :jr-j]
    assert np.all(dst < 256)
    return dst

def iquantify88(src:np.ndarray, quantification_table:np.ndarray):
    """
    input: src - 2d ndarray. quantification_table: 8x8 ndarray
    output: 2d np.float32 ndarray. (won't automatically pad)
    """
    assert quantification_table.shape == (8, 8)
    dst = np.zeros_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ir, jr = np.min((i+8, src.shape[0])), np.min((j+8, src.shape[1]))
            dst[i:i+8, j:j+8] = src[i:i+8, j:j+8] * quantification_table[:ir-i, :jr-j]
    return dst

def dc_delta88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size np.float32 ndarray
    perform delta encoding to DC(block88[0][0]) of each 8*8block (won't automatically pad or round)
    """
    src = src.astype(np.float32)
    dc_first, dc_old = src[0][0], src[0][0]
    # print(type(dc_first))
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            src[i][j] -= dc_old
            dc_old += src[i][j]
    src[0][0] = dc_first
    return src

def dc_idelta88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size np.float32 ndarray
    perform delta encoding to DC(block88[0][0]) of each 8*8block (won't automatically pad or round)
    """
    src = src.astype(np.float32)
    dc = src[0][0]
    src[0][0] = 0
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            dc += src[i][j]
            src[i][j]  = dc
    return src

def zigzag(src:np.ndarray, first_zig_down=False):
    """
    input: 2d ndarray
    first_zig_down: set the first movement to pos(0,0)->pos(1,0), default pos(0,0)->pos(0,1)
    output: same type 1d ndarray. zigzag code of src
    """
    up_side_down = src[::-1,:]
    sign = 1 if up_side_down.shape[0] % 2 == 0 else -1 # assue first_zig_right, make sure first (2*(k % 2)-1)*sign is +1
    sign *= -1 if first_zig_down else 1 # deal with first_zig_down occasion
    diags = [np.diagonal(up_side_down, k)[::(2*(k % 2)-1) * sign] for k in range(1-up_side_down.shape[0], up_side_down.shape[1])]
    return np.concatenate(diags)

def zigzag88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same type 1d ndarray
    perform zigzag to each 8*8block (won't automatically pad)
    """
    ret = []
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ret.append(zigzag(src[i:i+8, j:j+8]))
    return np.concatenate(ret)

def fill_diagonal_any(a:np.ndarray, offset:int, values:np.ndarray):
    if offset >= a.shape[1] or offset <= -a.shape[0]:
        raise ValueError("偏移量超出数组范围")
    start_row, start_col = (0, offset) if offset > 0 else (-offset, 0)
    diagonal_length = np.min((a.shape[0] - start_row, a.shape[1] - start_col))
    np.fill_diagonal(a[start_row:start_row+diagonal_length, start_col:start_col+diagonal_length], values[:diagonal_length])
    return a

def izigzag(src:np.ndarray, height:int, width:int, first_zig_down=False):
    """
    input: 1d ndarray. zigzag code
    output: same type 2d ndarray.
    """
    dst = np.zeros((height, width), dtype=src.dtype)
    i = 0
    for k in range(1-height, width):
        l = np.diagonal(dst, k).shape[0]
        sign = 1 if height % 2 == 0 else -1
        sign *= -1 if first_zig_down else 1
        fill_diagonal_any(dst, k, src[i:i+l][::(2*(k % 2)-1) * sign])
        i += l
    return dst[::-1,:]

def izigzag88(src:np.ndarray, height:int, width:int):
    """
    input: 1d ndarray
    output: same type 2d ndarray
    perform izigzag to zigzag code, 
    """
    dst = np.zeros((height, width), dtype=src.dtype)
    pos = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            ir, jr = np.min((i+8, height)), np.min((j+8, width))
            l = (ir - i) * (jr - j)
            dst[i:ir, j:jr] = izigzag(src[pos:pos+l], ir-i, jr-j)
            pos += l
    return dst


#########################
## dna process
#########################
def bytes2dna(src:np.ndarray):
    """
    input: 1d ndarray, dtype==np.uint8
    output: ACGT string
    perform 2bit map from bytes to dna sequence
    """
    mapper = list(sorted(["".join(x) for x in product("ACGT", "ACGT", "ACGT", "ACGT")]))
    return "".join([mapper[i] for i in src])

def dna2bytes(dna:str):
    """
    input: ACGT string
    output: 1d ndarray, dtype==np.uint8
    perform 2bit map from dna sequence to bytes
    """
    # Initialize an empty byte array
    encoded = bytearray()
    # Iterate over the DNA sequence in chunks of 4 bases
    for i in range(0, len(dna), 4):
        chunk = dna[i:i+4]
        # Convert each base to a 2-bit value using bitwise operations
        byte = 0
        for j, base in enumerate(chunk):
            if base == 'A':
                byte |= 0b00 << (6 - j*2)
            elif base == 'C':
                byte |= 0b01 << (6 - j*2)
            elif base == 'G':
                byte |= 0b10 << (6 - j*2)
            elif base == 'T':
                byte |= 0b11 << (6 - j*2)
        # Add the encoded byte to the byte array
        encoded.append(byte)
    return np.array(encoded, dtype=np.uint8)

def count_kmers(sequence:str, k:int, canonical=False):
    """
    output: collections.Counter. {kmer:count}
    """
    kmers = [sequence[i:i+k] if not canonical else 
             min(sequence[i:i+k], sequence[i:i+k].translate(str.maketrans("ATCG", "TAGC"))[::-1])
             for i in range(len(sequence)-k+1)]
    return Counter(kmers)


#########################
## test support functions
#########################
def timeit(func, *args, **kwargs):
    print(f"{(80-len(func.__name__))//2*'-'}{func.__name__}{(80-len(func.__name__))//2*'-'}")
    print("|||||arguments:", args, kwargs, "|||||")
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to run.")
    print("-"*80)
    return result

def assert_file_exist(filename):
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        sys.exit(1)

def center_crop(mat: np.ndarray, n: int) -> np.ndarray:
    """
    从 numpy 矩阵中截取前两维中心部分的子矩阵。
    
    Args:
    - mat: 输入的 numpy 矩阵。
    - n: 子矩阵的边长。
    
    Returns:
    - 截取后的子矩阵，也是一个 numpy 矩阵。
    """
    # 获取输入矩阵的高度和宽度
    h, w = mat.shape[:2]
    # 计算新矩阵的左上角和右下角坐标
    y1 = int((h - n) / 2)
    y2 = y1 + n
    x1 = int((w - n) / 2)
    x2 = x1 + n
    # 返回中心部分的子矩阵
    return mat[y1:y2, x1:x2]

def strnow(format:str):
    return datetime.datetime.now().strftime(format)




























if __name__ == "__main__":
    a=  zigzag(np.array([   [ 0,  1,  2,  3],
                            [ 4,  5,  6,  7],
                            [ 8,  9, 10, 11],]))
    b=  zigzag(np.array([   [ 0,  1,  2],
                            [ 4,  5,  6],
                            [ 8,  9, 10],
                            [12, 13, 14]]))
    c=  zigzag(np.array([   [ 0,  1,  2,  3],
                            [ 4,  5,  6,  7],
                            [ 8,  9, 10, 11],
                            [12, 13, 14, 15]]))
    d=  zigzag(np.array([   [ 0,  1,  2],
                            [ 4,  5,  6],
                            [ 8,  9, 10]]))
    print(a)
    print(b)
    print(c)
    print(d)
    print(izigzag(a, 3,4))
    print(izigzag(b, 4,3))
    print(izigzag(c, 4,4))
    print(izigzag(d, 3,3))

