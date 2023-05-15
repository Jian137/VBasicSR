from os import path as osp
from PIL import Image
import os
from basicsr.utils import scandir
from glob import glob
import numpy as np
def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = '/home/vipsl416-10-hujian/Datasets/BraTS18/BraTS_HR/flair/test_slice_video_split_2'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_BraTS18_test_split2_GT.txt'

    folder_list = sorted(os.listdir(gt_folder))

    with open(meta_info_txt, 'w') as f:
        for idx, foler_path in enumerate(folder_list):
            npy_list = glob(osp.join(gt_folder,foler_path,"*.npy"))
            #img = Image.open(osp.join(gt_folder, img_path))
            npy = np.load(npy_list[0])

            width, height, n_channel = npy.shape

            info = f'{foler_path} {len(npy_list)} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_div2k()
