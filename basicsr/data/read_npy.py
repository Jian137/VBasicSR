import numpy as np
import os
from os.path import join
file_name = '/home/vipsl416-4-zhanglize/Datasets/IXI/IXI_HR/T2/valid'
max_list = []

for file in os.listdir(file_name):
    np_file = np.load(join(file_name, file))
    # print(file)
    # print(np.max(np_file))
    max_list.append(np.max(np_file))
    # print(np.min(np_file))
    max_list.sort()

print(max_list)