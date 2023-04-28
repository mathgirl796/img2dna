from utils import *

def i2d_jpg(img:np.ndarray):
    """
    input: cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    output: str of dna sequence
    """
    img_shape = img.shape[:-1]
    # get jpeg zigzag raw
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32) - 128
    jpeg_raws = []
    for i, c in enumerate(cv2.split(yuv)):
        c = pad(c, base=8) # smoothly pad
        pad_shape = c.shape
        c = dct88(c)
        c = quantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table)
        c = c.round().clip(-128, 127) # 如果delta编码后再进行round，会导致量化误差的积累，图片越往右下误差越大
        c = dc_delta88(c)
        c = zigzag88(c)
        jpeg_raws.append(c.astype(np.uint8)) # 小心float到int的类型转换
    # convert to dna
    dna = "".join([bytes2dna(raw) for raw in jpeg_raws])
    # return
    return dna, img_shape, pad_shape

def d2i_jpg(dna:str, img_shape, pad_shape):
    """
    img_shape, pad_shape: (height, width)
    """
    jpeg_raws = dna2bytes(dna).reshape(3, -1)
    channels = []
    for i, raw in enumerate(jpeg_raws):
        raw = raw.astype(np.int8)
        c = izigzag88(raw, *pad_shape)
        c = dc_idelta88(c)
        c = iquantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table)
        c = idct88(c)
        c = ipad(c, *img_shape)
        channels.append(c)
    yuv = (np.dstack(channels) + 128).round().astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def i2d_jpg_64channels(img:np.ndarray):
    """
    input: cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    output: str of dna sequence
    """
    img_shape = img.shape[:-1]
    # get jpeg zigzag raw
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32) - 128
    jpeg_raws = []
    for i, c in enumerate(cv2.split(yuv)):
        c = pad(c, base=8) # smoothly pad
        pad_shape = c.shape
        c = dct88(c)
        c = quantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table)
        c = c.round().clip(-128, 127) # 如果delta编码后再进行round，会导致量化误差的积累，图片越往右下误差越大
        c = dc_delta88(c)
        c = zigzag88(c).reshape(-1, 64).T.reshape(-1)
        jpeg_raws.append(c.astype(np.uint8)) # 小心float到int的类型转换
    # convert to dna
    dna = "".join([bytes2dna(raw) for raw in jpeg_raws])
    # return
    return dna, img_shape, pad_shape

def d2i_jpg_64channels(dna:str, img_shape, pad_shape):
    """
    img_shape, pad_shape: (height, width)
    """
    jpeg_raws = dna2bytes(dna).reshape(3, -1)
    channels = []
    for i, raw in enumerate(jpeg_raws):
        c = raw.astype(np.int8)
        c = c.reshape(64, -1).T.reshape(-1,)
        c = izigzag88(c, *pad_shape)
        c = dc_idelta88(c)
        c = iquantify88(c, luminance_quantization_table if i == 0 else chrominance_quantization_table)
        c = idct88(c)
        c = ipad(c, *img_shape)
        channels.append(c)
    yuv = (np.dstack(channels) + 128).round().astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def i2d_yuv_1x1(img:np.ndarray):
    """
    input: cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    output: str of dna sequence
    """
    img_shape = img.shape[:-1]
    # get yuv raw
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    jpeg_raws = []
    for i, c in enumerate(cv2.split(yuv)):
        c = c.reshape(-1,)
        jpeg_raws.append(c)
    # convert to dna
    dna = "".join([bytes2dna(raw) for raw in jpeg_raws])
    # return
    return dna, img_shape, None

def d2i_yuv_1x1(dna:str, img_shape, pad_shape):
    """
    img_shape: (height, width)
    """
    jpeg_raws = dna2bytes(dna).reshape(3, -1)
    channels = []
    for i, c in enumerate(jpeg_raws):
        c = c.reshape(img_shape)
        channels.append(c)
    yuv = np.dstack(channels)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def i2d_yuv_8x8(img:np.ndarray):
    """
    input: cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
    output: str of dna sequence
    """
    img_shape = img.shape[:-1]
    # get yuv raw
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    jpeg_raws = []
    for i, c in enumerate(cv2.split(yuv)):
        c = pad(c, base=8)
        pad_shape = c.shape
        c = zigzag88(c)
        jpeg_raws.append(c)
    # convert to dna
    dna = "".join([bytes2dna(raw) for raw in jpeg_raws])
    # return
    return dna, img_shape, pad_shape

def d2i_yuv_8x8(dna:str, img_shape, pad_shape):
    """
    img_shape, pad_shape: (height, width)
    """
    jpeg_raws = dna2bytes(dna).reshape(3, -1)
    channels = []
    for i, c in enumerate(jpeg_raws):
        c = izigzag88(c, *pad_shape)
        c = ipad(c, *img_shape)
        channels.append(c)
    yuv = np.dstack(channels)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)