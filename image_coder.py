import numpy as np
import exnumpy as xnp
import excollections as xcl
import cv2
import collections
from skimage.metrics import structural_similarity
from skimage.data import cat
from exnumpy import zigzag_indices, izigzag_indices
from multiprocessing import Pool
from jpeg_component import *
from functools import partial
from itertools import repeat
from tqdm import tqdm
from utils import *

def im2zlines(im, bh, bw, qtable):
    h, w = im.shape
    ph, pw = h + (bh-h%bh)%bh, w + (bw-w%bw)%bw
    pad_im = np.pad(im, ((0, ph - h), (0, pw - w)), 'edge')
    blocks = pad_im.reshape(ph // bh, bh, pw // bw, bw).transpose(0, 2, 1, 3).reshape(-1, bh, bw)
    shift_blocks = (blocks.astype(np.int32) + np.iinfo(np.int8).min).astype(np.int8)
    dct_blocks = np.apply_along_axis(lambda x: cv2.dct(x.astype(np.float32).reshape(bh, bw)), axis=-1, arr=shift_blocks.reshape(-1, bh * bw))
    quan_blocks = np.around(dct_blocks / qtable).clip(-2047, 2047).astype(np.int32)
    zlines = quan_blocks.reshape(-1, bh * bw)[:, zigzag_indices(bh, bw)]
    return zlines

def zlines2im(zlines, bh, bw, qtable, h, w):
    ph, pw = h + (bh-h%bh)%bh, w + (bw-w%bw)%bw
    izigzag_lines = zlines.astype(np.float32)[:, izigzag_indices(bh ,bw)]
    dequan_blocks = izigzag_lines.reshape(-1, bh, bw) * qtable
    idct_blocks = np.apply_along_axis(lambda x: cv2.idct(x.reshape(bh, bw)), axis=-1, arr=dequan_blocks.reshape(-1, bh * bw)).squeeze()
    deshift_blocks = (idct_blocks - np.iinfo(np.int8).min).clip(0, 255).astype(np.uint8)
    reconstruct_im = deshift_blocks.reshape(ph // bh, pw // bw, bh, bw).transpose(0, 2, 1, 3).reshape(ph, pw)
    return reconstruct_im[:h, :w]

def zlines2rles(zlines, dc_huffman_table, ac_huffman_table, value2bin): 
    dc_values = zlines[:, 0]
    acs = zlines[:, 1:]
    ac_values = xnp.move_nonzeros_to_front(acs)
    ac_runs = xnp.count_consec_zeros(acs, filter_zero_len_consec_zeros=False)
    def zline2rle(dc_value, ac_value, ac_run, dc_huffman_table, ac_huffman_table, value2bin):
        rle = []
        # print(dc_value, ac_value, ac_run)
        dc_bin = value2bin(dc_value)
        rle.append(dc_huffman_table[len(dc_bin)] + dc_bin)
        for i in range(len(ac_value)):
            run = ac_run[i]
            value = ac_value[i]
            if value == 0:
                break
            while run >= 16:
                rle.append(ac_huffman_table[15 << 4 + 0])
                run -= 16
            value_bin = value2bin(value)
            rle.append(ac_huffman_table[(run << 4) + len(value_bin)] + value_bin) # 加法优先级高于移位，记得加括号
        return rle
    rles = []
    for dc_value, ac_value, ac_run in zip(dc_values, ac_values, ac_runs):
        rle = zline2rle(dc_value, ac_value, ac_run, dc_huffman_table, ac_huffman_table, value2bin)
        rles.append(rle)

    return rles # 列表的列表，子列表里是一些字符串，分别表示dc和各ac的huffman编码

def rle2zline(rle:str, reverse_dc_huffman_table, reverse_ac_huffman_table, bin2value):
    zline = []
    # dc
    pos = 0
    code = ''
    while pos < len(rle):
        code += rle[pos]
        pos += 1
        if code in reverse_dc_huffman_table:
            value_len = reverse_dc_huffman_table[code]
            value_bin = rle[pos:pos+value_len]
            value = bin2value(value_bin)
            zline.append(value)
            pos += value_len
            code = ''
            break
    # ac
    while pos < len(rle):
        code += rle[pos]
        pos += 1
        if code in reverse_ac_huffman_table:
            run = reverse_ac_huffman_table[code] >> 4
            value_len = reverse_ac_huffman_table[code] & 0x0f
            if run == 0 and value_len == 0: # EOB
                break
            value_bin = rle[pos:pos+value_len]
            value = bin2value(value_bin)
            zline.extend([0 for _ in range(run)])
            zline.append(value)
            pos += value_len
            code = ''
    return zline, pos

def rles2zlines(rles, reverse_dc_huffman_table:list, reverse_ac_huffman_table:list, bin2value): # rles: 字符串列表
    zlines = []
    for rle in rles:
        zlines.append(rle2zline(rle, reverse_dc_huffman_table, reverse_ac_huffman_table, bin2value)[0])
    return zlines # 列表列表，里面是解出来的dc、ac，不一定有多长


def test_libjpeg(path:str=None, params=[]):
    if isinstance(path, str):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.cvtColor(cat(), cv2.COLOR_RGB2GRAY)
    _, bytestream = cv2.imencode('.jpg', im, params)
    reconstruct_im = cv2.imdecode(bytestream, cv2.IMREAD_GRAYSCALE)
    ssim = structural_similarity(im, reconstruct_im)
    print(f'ssim: {ssim}, score: {(ssim - 0.84) / 0.16 * 50}')
    cv2.imshow('reconstruct_im', reconstruct_im)
    cv2.waitKey()

def test_image(path:str=None, Q=50, n=15):
    if isinstance(path, str):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.cvtColor(cat(), cv2.COLOR_RGB2GRAY)
    qtable = jpeg_lum_qtable(Q)
    zlines = im2zlines(im, 8, 8, qtable)

    # print(f'total row num: {len(zlines)}')
    # counter = collections.Counter(xnp.count_zeros_per_row(zlines))
    # print(f'zeros per row: {sorted(counter.items(), key=lambda x: x[0])}')

    # reconstruct_im = zlines2im(zlines, 8, 8, qtable, *im.shape)
    # ssim = structural_similarity(im, reconstruct_im)
    # print(f'Q: {Q}, ssim: {ssim}, score: {(ssim - 0.84) / 0.16 * 50} (full zline)')

    zlines = xnp.keep_n_leading_nonzero_elements(zlines, n)

    print(f'after keep n leading nonzeros')
    counter = collections.Counter(xnp.count_nonzero_per_row(zlines))
    print(f'nonzeros per row: {sorted(counter.items(), key=lambda x: x[0])}')
    counter = collections.Counter(xnp.count_trailing_zeros_per_row(zlines))
    print(f'trailing zeros per row: {sorted(counter.items(), key=lambda x: x[0])}')
    # consec_zeros = xnp.count_consec_zeros(zlines)
    # lead_consec_zeros = xnp.remove_n_trailing_nonzero_elements(consec_zeros, 1)
    # print(f'run zeros lengths: {xcl.count_sort(lead_consec_zeros.flatten().tolist())}')

    reconstruct_im = zlines2im(zlines, 8, 8, qtable, *im.shape)
    ssim = structural_similarity(im, reconstruct_im)
    print(f'Q: {Q}, ssim: {ssim}, score: {(ssim - 0.84) / 0.16 * 50} (keep {n} leading nonzero)')

    # cv2.imshow('reconstruct_im', reconstruct_im)
    # cv2.waitKey()

if __name__ == '__main__':

    image_list = ['image/small1.bmp', 'image/small2.bmp', 'image/small3.bmp', 
                  'image/crop1.bmp', 'image/crop2.bmp', 'image/crop3.bmp', 
                  'image/10DPI_1.bmp', 'image/10DPI_2.bmp', 'image/10DPI_3.bmp', 
                  'image/15DPI_1.bmp', 'image/15DPI_2.bmp', 'image/15DPI_3.bmp', 'image/15DPI_4.bmp', 
                  'image/20DPI_1.bmp', 'image/20DPI_2.bmp', 'image/20DPI_3.bmp', ]
    
    Q = 95
    im = cv2.imread(image_list[2], cv2.IMREAD_GRAYSCALE)
    zlines = timeit(im2zlines, im, 8, 8, jpeg_lum_qtable(Q))
    rles = timeit(zlines2rles, zlines, jpeg_dc_huffman_table(), jpeg_ac_huffman_table(), value2bin=jpeg_value2bin)
    rrles = [''.join(x) for x in rles]
    zzlines = rles2zlines(rrles, jpeg_dc_huffman_table(reverse=True), jpeg_ac_huffman_table(reverse=True), jpeg_bin2value)
    pad_zzlines = np.zeros_like(zlines)
    for i, zzline in enumerate(zzlines):
        pad_zzlines[i, :len(zzline[:pad_zzlines.shape[1]])] = zzline[:pad_zzlines.shape[1]]
    reconstruct_im = zlines2im(pad_zzlines, 8, 8, jpeg_lum_qtable(Q), *im.shape)

    print(f'ssim: {structural_similarity(im, reconstruct_im)}')
    for i in range(len(zlines)):
        print(zlines[i].tolist())
        print(rles[i])
        print(zzlines[i])
        input()