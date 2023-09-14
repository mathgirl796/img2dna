import numpy as np

# 将此函数返回值ret作用于矩阵arr(即arr[ret])，可得到arr的zigzag排列
def zigzag_indices(h, w, first_zig_down=False) -> list:
    src = np.arange(h * w, dtype=np.int32).reshape(h, w)
    up_side_down = src[::-1,:]
    sign = 1 if up_side_down.shape[0] % 2 == 0 else -1 # assue first_zig_right, make sure first (2*(k % 2)-1)*sign is +1
    sign *= -1 if first_zig_down else 1 # deal with first_zig_down occasion
    diags = [np.diagonal(up_side_down, k)[::(2*(k % 2)-1) * sign] for k in range(1-up_side_down.shape[0], up_side_down.shape[1])]
    return np.concatenate(diags).tolist()

# 将此函数返回值ret作用于矩阵arr的zigzag排列(即zigzag[ret])，可恢复arr(恢复成1维，需要自己reshape)
def izigzag_indices(h, w, first_zig_down=False) -> list:
    src = zigzag_indices(h, w, first_zig_down=first_zig_down)
    return [src.index(i) for i in range(h * w)]

# 返回每一行0的个数
def count_zeros_per_row(arr: np.ndarray) -> np.ndarray:
    zeros_per_row = np.sum(arr == 0, axis=1)
    return zeros_per_row

# 返回每一行尾部连续0的个数
def count_trailing_zeros_per_row(arr: np.ndarray) -> np.ndarray:
    mask = arr[:, ::-1] == 0  # 反转数组，并与0进行比较，生成布尔掩码
    trailing_zeros_per_row = np.argmax(~mask, axis=1)  # 找到每行中最后一个非零元素的索引
    return trailing_zeros_per_row

# 返回每一行非零元素的个数
def count_nonzero_per_row(arr: np.ndarray) -> np.ndarray:
    nonzero_per_row = np.count_nonzero(arr, axis=1)
    return nonzero_per_row

# 将每一行最前的n个非零元素保留，其余置为0
def keep_n_leading_nonzero_elements(arr: np.ndarray, n) -> np.ndarray:
    mask = arr != 0  # 创建非零元素的掩码
    nonzero_indices = np.cumsum(mask, axis=1) <= n  # 获取每行前n个非零元素的索引
    result = np.where(nonzero_indices, arr, 0)  # 将非零元素之外的元素置为0
    return result

# 将每一行最后的n个非零元素置为0
def remove_n_trailing_nonzero_elements(arr: np.ndarray, n) -> np.ndarray:
    mask = arr != 0  # 创建非零元素的掩码
    nonzero_indices = np.cumsum(mask, axis=1) > (np.cumsum(mask, axis=1).max(axis=1, keepdims=True) - n)  # 获取每行排除最后n个非零元素的索引
    result = np.where(nonzero_indices, 0, arr)  # 将非零元素之外的元素置为0
    return result

# 计算每行每个连续0的长度，包括末尾的连续0。关闭filter：两个连续非0元素间会记录一个长为0的连续0,包括一行的最前面
def count_consec_zeros(arr: np.ndarray, filter_zero_len_consec_zeros=True) -> np.ndarray:
    result = np.zeros_like(arr, dtype=int) # 创建一个与输入数组形状相同的全零数组，用于存储结果
    count, length = np.zeros(arr.shape[0], dtype=int), np.zeros(arr.shape[0], dtype=int) # 初始化计数器和当前零串长度
    for j in range(arr.shape[1]): # 遍历每一列
        grow_pos = arr[:, j] == 0
        save_pos = (arr[:, j] != 0) & (length > 0) if filter_zero_len_consec_zeros else (arr[:, j] != 0)
        length[grow_pos] += 1
        result[save_pos, count[save_pos]] = length[save_pos]
        count[save_pos] += 1
        length[save_pos] = 0
    save_pos = length > 0
    result[save_pos, count[save_pos]] = length[save_pos] # 处理行末尾是零串的情况
    return result

# 删除最后一个非全0列之后的所有列
def cut_trailing_zero_cols(arr: np.ndarray) -> np.ndarray:
    last_nonzero_col = np.max(np.where(np.any(arr, axis=0))) # 找到最后一个非全0列的索引
    new_arr = arr[:, :last_nonzero_col+1] 
    return new_arr

# 删除每一行的尾部连续0，返回二级列表
def cut_trailing_zeros_per_row(arr: np.ndarray) -> list:
    return [np.trim_zeros(x, 'b').tolist() for x in arr]

# 将每行的非零元素按顺序移到最前面
def move_nonzeros_to_front(arr) -> np.ndarray:
    arr = np.apply_along_axis(lambda x: np.pad(x[x!=0], (0, x.size-np.count_nonzero(x))), axis=1, arr=arr)
    return arr


if __name__ == '__main__':
    a = np.array([  
        [2, 0, 0, 3, 0],
        [0, 4, 4, 0, 0],
        [0, 0, 0, 0, 5],
        [1, 2, 3, 4, 5]])
    b = count_consec_zeros(a, False)
    c = move_nonzeros_to_front(a)
    print(b)
    print(c)
    print(cut_trailing_zeros_per_row(c))
    