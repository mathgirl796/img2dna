import numpy as np


def rles2zlines(rle):
    pass

def jpeg_value2bin(value):
    if value == 0:
        return ''
    elif value > 0:
        return bin(value)[2:]
    else:
        return '0' + bin(abs(value))[3:]
    
def jpeg_bin2value(value_bin):
    if value_bin == '':
        return 0
    elif len(value_bin) == 1:
        return ((-1) ** (1 - int(value_bin[0])))
    else:
        return ((-1) ** (1 - int(value_bin[0]))) * int('1' + value_bin[1:], base=2)

def jpeg_lum_qtable(Q=50):
    Q = np.clip(Q, 1, 100)
    std_jpeg_lum_qtable = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], 
                   [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
    if Q >= 50: 
        qtable = np.around((2 - Q / 50) * std_jpeg_lum_qtable).clip(1, None)
    else:
        qtable = np.around((50 / Q) * std_jpeg_lum_qtable)
    # print(qtable)
    return qtable

def jpeg_dc_huffman_table(reverse=False, dc_nrcodes = [0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dc_values = [4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11]):
    # DC哈夫曼编码表
    standard_dc_nrcodes = dc_nrcodes # 默认情况下，码长为3的有七个，从小到大分别编码数值4/5/3/2/6/1/0
    standard_dc_values = dc_values
    pos_in_table = 0
    code_value = 0
    dc_huffman_table = {} # 默认情况下，table[4]得到二进制数b'000', table[5] = '001'
    for i in range(1, 1 + len(dc_nrcodes)):
        for j in range(1, standard_dc_nrcodes[i - 1] + 1):
            if not reverse:
                dc_huffman_table[standard_dc_values[pos_in_table]] = bin(code_value)[2:].rjust(i, '0')
            else:
                dc_huffman_table[bin(code_value)[2:].rjust(i, '0')] = standard_dc_values[pos_in_table]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1
    return dc_huffman_table

def jpeg_ac_huffman_table(reverse=False, ac_nrcodes = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], ac_values = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa]):
    # AC哈夫曼编码表
    standard_ac_nrcodes = ac_nrcodes
    standard_ac_values = ac_values # 0xAB中，A表示run，B表示huffman码长. 默认情况下，共有162种码，编码[0x00,0xfa]之间的某个数
    # print(sum(standard_ac_nrcodes), len(standard_ac_values))
    pos_in_table = 0
    code_value = 0
    ac_huffman_table = {}
    for i in range(1, 1 + len(ac_nrcodes)):
        for j in range(1, standard_ac_nrcodes[i - 1] + 1):
            if not reverse:
                ac_huffman_table[standard_ac_values[pos_in_table]] = bin(code_value)[2:].rjust(i, '0')
            else:
                ac_huffman_table[bin(code_value)[2:].rjust(i, '0')] = standard_ac_values[pos_in_table]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1
    return ac_huffman_table
