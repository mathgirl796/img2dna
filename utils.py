import numpy as np
import cv2
from itertools import product



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
    output: same size and dtype ndarray
    perform dct to each 8*8block (won't automatically pad)
    """
    dst = np.zeros_like(src, dtype=src.dtype)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            dst[i:i+8, j:j+8] = cv2.dct(src[i:i+8, j:j+8])
    return dst

def idct88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size and dtype ndarray
    perform idct to each 8*8block (won't automatically pad)
    """
    dst = np.zeros_like(src, dtype=src.dtype)
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

def quantify88(src:np.ndarray, quantification_table:np.ndarray, quantify_type=np.uint8):
    """
    input: src - 2d ndarray. quantification_table: 8x8 ndarray
    output: 2d quantify_type ndarray. (won't automatically pad)
    """
    assert quantification_table.shape == (8, 8)
    dst = np.zeros_like(src, dtype=np.float32)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ir, jr = np.min((i+8, src.shape[0])), np.min((j+8, src.shape[1]))
            dst[i:i+8, j:j+8] = src[i:i+8, j:j+8] / quantification_table[:ir-i, :jr-j]
    assert np.all(dst < 256)
    return np.around(dst).astype(quantify_type)

def iquantify88(src:np.ndarray, quantification_table:np.ndarray, iquantify_type=np.float32):
    """
    input: src - 2d ndarray. quantification_table: 8x8 ndarray
    output: 2d iquantify_type ndarray. (won't automatically pad)
    """
    assert quantification_table.shape == (8, 8)
    dst = np.zeros_like(src, dtype=iquantify_type)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ir, jr = np.min((i+8, src.shape[0])), np.min((j+8, src.shape[1]))
            dst[i:i+8, j:j+8] = src[i:i+8, j:j+8] * quantification_table[:ir-i, :jr-j]
    return dst

def dc_delta88(src:np.ndarray):
    """
    input: 2d ndarray
    output: same size dtype ndarray
    perform delta encoding to DC(block88[0][0]) of each 8*8block (won't automatically pad)
    """
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
    output: same size dtype ndarray
    perform delta encoding to DC(block88[0][0]) of each 8*8block (won't automatically pad)
    """
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
    output: 1d ndarray. zigzag code of src
    """
    up_side_down = src[::-1,:]
    sign = 1 if up_side_down.shape[0] % 2 == 0 else -1 # assue first_zig_right, make sure first (2*(k % 2)-1)*sign is +1
    sign *= -1 if first_zig_down else 1 # deal with first_zig_down occasion
    diags = [np.diagonal(up_side_down, k)[::(2*(k % 2)-1) * sign] for k in range(1-up_side_down.shape[0], up_side_down.shape[1])]
    return np.concatenate(diags)

def izigzag(src:np.ndarray, height:int, width:int, first_zig_down=False):
    """
    input: 1d ndarray. zigzag code
    output: 2d ndarray.
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

def zigzag88(src:np.ndarray):
    """
    input: 2d ndarray
    output: 1d ndarray
    perform zigzag to each 8*8block (won't automatically pad)
    """
    ret = np.array([], dtype=src.dtype)
    for i in range(0, src.shape[0], 8):
        for j in range(0, src.shape[1], 8):
            ret = np.concatenate((ret, zigzag(src[i:i+8, j:j+8])))
    return ret

def izigzag88(src:np.ndarray, height:int, width:int):
    """
    input: 1d ndarray
    output: 2d ndarray
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

def bytes2dna(src:np.ndarray):
    """
    input: 1d ndarray, dtype==np.int8
    output: ACGT string
    perform 2bit map from bytes to dna sequence
    """
    assert src.dtype == np.int8
    mapper = list(sorted(["".join(x) for x in product("ACGT", "ACGT", "ACGT", "ACGT")]))
    return "".join([mapper[i] for i in src])

def dna2bytes(dna:str):
    """
    input: ACGT string
    output: 1d ndarray, dtype==np.int8
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
    return np.array(encoded, dtype=np.int8)

def fill_diagonal_any(a:np.ndarray, offset:int, values:np.ndarray):
    if offset >= a.shape[1] or offset <= -a.shape[0]:
        raise ValueError("偏移量超出数组范围")
    start_row, start_col = (0, offset) if offset > 0 else (-offset, 0)
    diagonal_length = np.min((a.shape[0] - start_row, a.shape[1] - start_col))
    np.fill_diagonal(a[start_row:start_row+diagonal_length, start_col:start_col+diagonal_length], values[:diagonal_length])
    return a

def split_channels(src:np.ndarray):
    """
    input: 3d ndarray
    output: n 2d ndarrays, n=input.shape[-1]
    """
    return (src[:,:,i] for i in range(src.shape[-1]))

def y2bgr(src:np.ndarray):
    """
    input: 2d ndarray
    output: 3d ndarray. shape=(src.shape[0], src.shape[1], 3)
    """
    fill = np.full(src.shape, 0, dtype=src.dtype)
    return cv2.cvtColor(np.dstack((src, fill, fill)), cv2.COLOR_YUV2BGR)

def u2bgr(src:np.ndarray):
    """
    input: 2d ndarray
    output: 3d ndarray. shape=(src.shape[0], src.shape[1], 3)
    """
    fill = np.full(src.shape, 0, dtype=src.dtype)
    return cv2.cvtColor(np.dstack((fill, src, fill)), cv2.COLOR_YUV2BGR)

def v2bgr(src:np.ndarray):
    """
    input: 2d ndarray
    output: 3d ndarray. shape=(src.shape[0], src.shape[1], 3)
    """
    fill = np.full(src.shape, 0, dtype=src.dtype)
    return cv2.cvtColor(np.dstack((fill, fill, src)), cv2.COLOR_YUV2BGR)





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