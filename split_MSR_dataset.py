import os
from glob import glob
import shutil
from tqdm import tqdm
gt_root = "/mnt/zlz/Dataset/BraTS18/BraTS_HR/flair/test_slice_video"
gt_dest = "/mnt/zlz/Dataset/BraTS18/BraTS_HR/flair/test_slice_video_split_2"
lq_root = "/mnt/zlz/Dataset/BraTS18/BraTS_LR/bicubic_2x/flair/test_slice_video"
lq_dest = "/mnt/zlz/Dataset/BraTS18/BraTS_LR/bicubic_2x/flair/test_slice_video_split_2"

keys = os.listdir(gt_root)

if not os.path.exists(gt_dest):
    os.mkdir(gt_dest)
if not os.path.exists(lq_dest):
    os.mkdir(lq_dest)

for key in tqdm(keys):
    filenames = sorted(os.listdir(os.path.join(gt_root,key)))
    for i,filename in enumerate(filenames):
        if i<len(filenames)/2:
            gt_folder = os.path.join(gt_dest,key+"_1")
            lq_folder = os.path.join(lq_dest,key+"_1")
            if not os.path.exists(gt_folder):
                os.mkdir(gt_folder)
            if not os.path.exists(lq_folder):
                os.mkdir(lq_folder)
            shutil.copy(os.path.join(gt_root,key,filename),os.path.join(gt_folder,filename))
            shutil.copy(os.path.join(lq_root,key,filename),os.path.join(lq_folder,filename))
        else:
            gt_folder = os.path.join(gt_dest,key+"_2")
            lq_folder = os.path.join(lq_dest,key+"_2")
            if not os.path.exists(gt_folder):
                os.mkdir(gt_folder)
            if not os.path.exists(lq_folder):
                os.mkdir(lq_folder)
            shutil.copy(os.path.join(gt_root,key,filename),os.path.join(gt_folder,str(i-int(len(filenames)/2)).zfill(8)+".npy"))
            shutil.copy(os.path.join(lq_root,key,filename),os.path.join(lq_folder,str(i-int(len(filenames)/2)).zfill(8)+".npy"))



