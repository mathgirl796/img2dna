__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"


# python builtin packages
import traceback
import subprocess
import sys
import time
import collections
import itertools
import tempfile
import multiprocessing
import functools

# competition allowed packages
import numpy as np
import cv2
from PIL import Image
from Chamaeleo.utils import indexer

# competition package
from evaluation import DefaultCoder, EvaluationPipeline

# other packages
from tqdm import tqdm, trange

# personal packages
from image_coder import *
from exnumpy import *
from excollections import *
from align import *


class Coder(DefaultCoder):

    def __init__(self, team_id: str = "none"):
        super().__init__(team_id=team_id)
        self.index_binary_length = 19
        self.Q = 60
        self.logical_depth = 17

    def image_to_dna(self, input_image_path, need_logs=True):

        print(f'start image_to_dna, time = {time.strftime("%Y-%m-%d, %H:%M:%S")}')
        print(f'init tables, Q = {self.Q}, logical_depth = {self.logical_depth}')
        
        self.lum_qtable = jpeg_lum_qtable(self.Q)
        self.reverse_dc_huffman_table = jpeg_dc_huffman_table(reverse=True)
        self.reverse_ac_huffman_table = jpeg_ac_huffman_table(reverse=True)

        print(f'read image from {input_image_path}')
        im = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        ph, pw = h + (8-h%8)%8, w + (8-w%8)%8
        im = np.pad(im, [(0, ph - h), (0, pw - w)], mode='edge')
        self.real_h, self.real_w = h, w
        self.h, self.w = ph, pw


        print('convert image to zlines')
        for i in trange(1):
            zlines = im2zlines(im, 8, 8, self.lum_qtable)
            rles = zlines2rles(zlines, jpeg_dc_huffman_table(), jpeg_ac_huffman_table(), jpeg_value2bin)
        print('zlines_trailing_zero distribution:', count_sort(count_trailing_zeros_per_row(zlines)))
        print('rle_bit_num distribution:', count_sort([len(''.join(x)) for x in rles]))

        print('build binary strings')
        with tqdm(total=len(rles)) as pbar:
            bstr = ''
            is_full = False
            yue = False
            bstr_list = []
            bstr_rle_num_list = [] # debug
            idx = 0
            while idx < len(rles):
                rle = rles[idx]
                # 组装一个新的二进制串
                if len(bstr) == 0: 
                    bstr += bin(idx)[2:].rjust(self.index_binary_length, '0')
                    bstr_rle_num_list.append(0)
                    # 试图把它喂饱
                    for item in rle: 
                        if len(bstr) + len(item) + 4 <= 200:
                            bstr += item
                            continue
                        else:
                            is_full = True
                            break
                    # 喂没喂饱这一块都结了，后面加上'1010'
                    bstr += '1010'
                    bstr_rle_num_list[-1] += 1
                # 继续喂一个已有的二进制串
                else:
                    lens = [len(x) for x in rle]
                    # 要吐了，不喂了
                    if len(bstr) + sum(lens) + 4 > 200:
                        is_full = True
                        yue = True
                    # 吃了也不会吐，所以吃了
                    else:
                        bstr += ''.join(rle)
                        bstr += '1010'
                        bstr_rle_num_list[-1] += 1
                # 喂饱了，保存，开始编码新的二进制串
                if is_full or idx == len(rles) - 1: 
                    while len(bstr) < 100:
                        bstr += '1'
                    bstr_list.append(bstr)
                    bstr = ''
                    is_full = False
                if yue:
                    yue = False
                    idx -= 1
                else:
                    pbar.update(1)
                idx += 1
                
        print('bstr_rle_num distribution:', count_sort(bstr_rle_num_list))

        print('encode dna string')
        dna_list = []
        for bstr in tqdm(bstr_list):
            dna = ''.join([['AT', 'CG'][idx % 2][int(ch)] for idx, ch in enumerate(bstr)])
            dna_list.append(dna)
            
        print(f'seq_num / block_num: {len(dna_list)}/{self.h * self.w / 8 / 8} = {len(dna_list) / (self.h * self.w / 8 / 8)}')
        print(f'seq_len distribution: {count_sort([len(x) for x in dna_list])}')

        print('make logical redundancy')
        dna_list *= self.logical_depth

        return dna_list

    def dna_to_image(self, dna_sequences, output_image_path, need_logs=True):

        print(f'start dna_to_image, time = {time.strftime("%Y-%m-%d, %H:%M:%S")}')

        print('cluster dna seqs')
        dna_clusters = collections.defaultdict(list)
        for idx, seq in tqdm(enumerate(dna_sequences), total=len(dna_sequences)):
            dna_clusters[seq[:self.index_binary_length]].append(seq)
        expect_cluster_num = len(dna_sequences) // self.logical_depth
        ordered_clusters = sorted(dna_clusters.values(), key=lambda x: len(x), reverse=True)
        good_clusters = ordered_clusters[:expect_cluster_num]
        bad_clusters = ordered_clusters[expect_cluster_num:]
        print(f'good_cluster num: {len(good_clusters)}, bad_cluster num: {len(bad_clusters)}')
        print(f'good_cluster total seq num: {sum(len(x) for x in good_clusters)}, bad_cluster total seq num: {sum(len(x) for x in bad_clusters)}')
        print(f'good_cluster_size distribution:', count_sort([len(x) for x in good_clusters]))
        print(f'bad_cluster_size distribution:', count_sort([len(x) for x in bad_clusters]))
        
        print('make consensus')
        if True:
            con_list = []
            import pyabpoa as pa
            msa_aligner = pa.msa_aligner(aln_mode='g')
            for cluster in tqdm(good_clusters):
                con = msa_aligner.msa(cluster, True, False, max_n_cons=1).cons_seq[0]
                con_list.append(con)
        else:
            pool = multiprocessing.Pool()
            con_list = pool.map(consensus_abpoa, good_clusters)
            pool.close()
            pool.join()
        
        print('decode huffman')
        zline_dict = {}
        for idx, con in tqdm(enumerate(con_list), total=len(con_list)):
            bstr = con.replace('A', '0').replace('T', '1').replace('C', '0').replace('G', '1')
            index, rle = int(bstr[:self.index_binary_length], base=2), bstr[self.index_binary_length:]
            if index >= self.h * self.w // 8 // 8:
                continue

            while len(rle) > 0:
                zline, pos = rle2zline(rle, self.reverse_dc_huffman_table, self.reverse_ac_huffman_table, jpeg_bin2value)
                if zline == []:
                    break

                zline_dict[index] = zline
                rle = rle[pos:]
                index += 1

        print('inflate zline list rows')
        min_dc = min(zline_dict.values(), key=lambda x: x[0])[0]
        print(f'zline_dict item num: {len(zline_dict)}')
        zline_list = []
        for index in tqdm(range(self.h * self.w // 8 // 8), total=self.h * self.w // 8 // 8):
            if index in zline_dict:
                zline_list.append(zline_dict[index])
            else:
                zline_list.append([min_dc])

        print('inflate zline list columns')
        zlines = np.zeros((self.h * self.w // 8 // 8, 64))
        for idx, zline in tqdm(enumerate(zline_list), total=len(zline_list)):
            zlines[idx, :len(zline[:zlines.shape[1]])] = zline[:zlines.shape[1]]

        print('convert zlines to image')
        im = zlines2im(zlines, 8, 8, self.lum_qtable, self.h, self.w)
        im = im[:self.real_h, :self.real_w]

        cv2.imwrite(output_image_path, im)
        
        return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_id', type=int, metavar='', required=False, default=6)
    parser.add_argument('-q', '--quality', type=int, metavar='', required=False, default=60)
    parser.add_argument('-d', '--depth', type=int, metavar='', required=False, default=17)
    args = parser.parse_args()

    coder = Coder()
    coder.Q = args.quality
    coder.logical_depth = args.depth
    image_list = ['image/10DPI_1.bmp', 'image/10DPI_2.bmp', 'image/10DPI_3.bmp', 
                  'image/15DPI_1.bmp', 'image/15DPI_2.bmp', 'image/15DPI_3.bmp', 'image/15DPI_4.bmp', 
                  'image/20DPI_1.bmp', 'image/20DPI_2.bmp', 'image/20DPI_3.bmp', 
                  'image/small1.bmp', 'image/small2.bmp', 'image/small3.bmp', 
                  'image/crop1.bmp', 'image/crop2.bmp', 'image/crop3.bmp',]
    
    try:    
        start_time = time.time()
        print(f"""[main] start""")

        output_dir = 'output'
        subprocess.run(f'mkdir -p {output_dir}', shell=True)

        image_id = args.image_id
        pipeline = EvaluationPipeline(coder=coder, error_free=False)
        pipeline(input_image_path=image_list[image_id], output_image_path=f"{output_dir}/{image_id}_out.bmp",
                source_dna_path=f"{output_dir}/{image_id}_encoded.fasta", target_dna_path=f"{output_dir}/{image_id}_sequenced.fasta", random_seed=2023)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        end_time = time.time()
        print(f"""[main] used {end_time - start_time} secs""")
