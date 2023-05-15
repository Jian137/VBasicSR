from glob import glob
from tqdm import tqdm
import numpy as np
files = glob("/home/vipsl416-10-hujian/Datasets/BraTS18/*/*/*/*/*/*.npy")

for file in tqdm(files):
    npy = np.load(file)
    if len(npy.shape)==2:
        npy = np.expand_dims(npy,axis=2)
        np.save(file,npy)