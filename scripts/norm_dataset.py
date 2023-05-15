import os
import numpy as np
from os.path import join
from tqdm import tqdm

# video norm
# finished : all HR
# LR:bicubic_2x、bicubic_3x、 bicubic_4x

# slice norm
# finished : ALL HR
# LR:bicubic_2x、bicubic_3x、bicubic_4x

data_path = '/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_LR/bicubic_4x'

statement_path = '/home/vipsl416-4-zhanglize/MSR/SR Code/BasicSR/results/modal_statement'
modals = ["PD", "T1", "T2"]
status = ["train", "valid", "test"]

# run once

def trans_to_normalize():
    for modal in modals:
        for stat in status:
            cur_stat_path = join(join(statement_path, modal), stat)
            cur_slice_dir = join(join(data_path, modal), stat) + "_slice_video_norm"
            stat_file = open(cur_stat_path)
            for line in stat_file:     # 读取文件的每一行
                cur_num, cur_max = line.split(" ")[0], line.split(" ")[1]
                cur_max = int(cur_max)
                cur_dir = join(cur_slice_dir, cur_num)
                # print(cur_slice_dir)
                print(cur_dir)
                for npy_slice in os.listdir(cur_dir):  # 当前文件夹的所有slice都要除以max
                    npy_file = np.load(join(cur_dir, npy_slice))
                    print(np.max(npy_file))
                    norm_npy_file = np.float32(npy_file) / np.float32(cur_max)
                    np.save(join(cur_dir, npy_slice), norm_npy_file)   # 将npy文件保存

# run once
def trans_tonormalize_slice():
    for modal in modals:
        for stat in status:
            cur_stat_path = join(join(statement_path, modal), stat)
            cur_slice_dir = join(join(data_path, modal), stat) + "_slice"
            stat_file = open(cur_stat_path)
            for line in tqdm(stat_file):
                cur_num, cur_max = line.split(" ")[0], line.split(" ")[1]
                cur_max = int(cur_max)
                for npy_slice in os.listdir(cur_slice_dir):
                    volume, slice = npy_slice.split("_")[-2].zfill(3), npy_slice.split("_")[-1].split(".")[0].zfill(3)
                    if volume == cur_num:
                        # print(volume)
                        # print(cur_num)
                        npy_file = np.load(join(cur_slice_dir, npy_slice))
                        # print(np.max(npy_file))
                        # print("\n")
                        norm_npy_file = np.float32(npy_file) / np.float32(cur_max)
                        np.save(join(cur_slice_dir, npy_slice), norm_npy_file)   # 将npy文件保存



def trans_to_digits():
    for modal in modals:
        for stat in status:
            cur_slice_dir = join(join(data_path, modal), stat) + "_slice"
            for file in tqdm(os.listdir(cur_slice_dir)):
                volume, slice = file.split("_")[-2], file.split("_")[-1].split(".")[0]
                new_name = volume.zfill(3) + "_" + slice.zfill(3) + ".npy"
                # print(file)
                # print(new_name)
                os.rename(join(cur_slice_dir, file), join(cur_slice_dir, new_name))


def trans_npy_nomal():
    pass   #TODO


if __name__ == "__main__":
    # trans_to_normalize()
    trans_tonormalize_slice()
    print("------------norm_dataset_test------------")


