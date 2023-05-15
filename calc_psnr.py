import os
import numpy as np
import torch


# 读取file1 和 file 2 的数据，并且计算psnr, ssim，存到一个字典里面
# 使用: 第一项得到一个字典，里面有着每个数据对应的指标，第二项为平均psnr，第三项为平均ssim

def calc_psnr(folder1: str, folder2: str):
    clac_psnr = calculate_psnr_pt # 把对应函数替换过来即可
    calc_ssim = calculate_ssim_pt
    calc_result = {}
    sum = torch.zeros(2)
    f1_list = os.listdir(folder1)
    f1_list.sort()
    f2_list = os.listdir(folder1)
    assert(len(f1_list) == len(f2_list))
    for file in f1_list:
        f1_file = os.path.join(folder1, file)
        f2_file = os.path.join(folder2, file)
        assert(os.path.exists(f2_file))  # 判断是否有这个文件
        file1_npy = np.load(f1_file).astype("float32")
        file2_npy = np.load(f2_file).astype("float32")
        file1_tensor = torch.as_tensor(file1_npy)
        file2_tensor = torch.as_tensor(file2_npy)
        calc_result[file] = torch.tensor([calc_psnr(file1_npy, file2_npy), calc_ssim(file1_npy, file2_npy)])

    for query in calc_result.values():
        sum += query

    return calc_result, sum[0] / len(f1_list), sum[1] / len(f1_list)



