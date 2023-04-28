import cv2
from apps import *

import warnings
import sys
import contextlib
import datetime

# 遇到运行时警告直接退出程序
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

def test_encode_decode(file_name, crop_side_len=None):
    assert_file_exist(file_name)
    img=cv2.imread(file_name, cv2.IMREAD_COLOR)
    if crop_side_len:
        img = center_crop(img, crop_side_len)
    print(f"img.shape:{img.shape}")
    dna, img_shape, pad_shape = img2dna(img)
    c = dna2img(dna, img_shape, pad_shape)
    err = np.abs(img.astype(np.float32) - c.astype(np.float32))
    print("loss", np.sum(err)/err.size)
    cv2.imshow(file_name, img)
    cv2.imshow("rebuild", c)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def test_count_kmer(file_name, kmer_len=256, crop_side_len=512, save_dna_txt=False):
    assert_file_exist(file_name)
    # 读取文件
    img=cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = center_crop(img, crop_side_len)
    print(f"img.shape:{img.shape}")
    dna, img_shape, pad_shape = img2dna(img)
    if save_dna_txt:
        with open(f"{file_name}.txt", "w", encoding="utf8") as f:
            f.write(dna)
    print(f"dna_len:{len(dna)}, base_count:{Counter(dna)}")
    counter = count_kmers(dna, kmer_len)
    print("unique kmer: ", len(counter))
    return counter

def test_dunhuang(kmer_len=256, crop_side_len=512):
    counters = []
    for i in range(1, 47):
        img_path = f"p.1071/img/T{i:07d}.jpg"
        c = timeit(test_count_kmer, img_path, kmer_len=kmer_len, crop_side_len=crop_side_len)
        counters.append(c)  
        if i == 1:
            icounter = c
        else:
            icounter = icounter & c
        print(f"processed {i} files, intersect unique kmer num is {len(icounter)}")
    average_ukmer_num = sum([len(c) for c in counters]) / len(counters)
    print(f"average unique kmer num for each img is {average_ukmer_num}")


# register coder here
coders = {
    0: [i2d_jpg, d2i_jpg],
    1: [i2d_jpg_64channels, d2i_jpg_64channels],
    2: [i2d_yuv_1x1, d2i_yuv_1x1],
    3: [i2d_yuv_8x8, d2i_yuv_8x8],
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # timeit(test_encode_decode, "demo_data/len_std.jpg")
        # timeit(test_encode_decode, "demo_data/color_pure.png")
        # timeit(test_encode_decode, "p.1071\T0000003.jpg", crop_side_len=512)
        exit(1)

    # parse arguments
    print(sys.argv)
    select_coder = int(sys.argv[1])
    kmer_len = int(sys.argv[2])

    # test
    img2dna = coders[select_coder][0]
    dna2img = coders[select_coder][1]
    with open(f"p.1071/log/log_{strnow('%Y-%m-%d-%H-%M-%S')}.txt", "w", encoding="utf8") as flog:
        with contextlib.redirect_stdout(flog):
            print(f"select_coder: {img2dna.__name__}\t{dna2img.__name__}")
            timeit(test_dunhuang, kmer_len=kmer_len, crop_side_len=512)
