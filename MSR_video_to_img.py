import os
import shutil
from glob import glob
from tqdm import tqdm

gt_root = ""
gt_dst = ""
lq_root = ""
lq_dst = ""

keys = os.listdir(gt_root)

if not os.path.exists(gt_dst):
    os.mkdir(gt_dst)
if not os.path.exists(lq_dst):
    os.mkdir(lq_dst)


for key in tqdm(keys):
    filenames = sorted(os.listdir(os.path.join(gt_root,key)))
    for i,filename in enumerate(filenames):




        shutil.copy(os.path.join(gt_root,key,filename),os.path.join(gt_dst,key+"_"+filename))
        shutil.copy(os.path.join(lq_root,key,filename),os.path.join(lq_dst,key+"_"+filename))
