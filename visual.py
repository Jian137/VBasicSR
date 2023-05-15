"""
可视化tensor代码

"""
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/home/vipsl416-4-zhanglize/MSR/SR Code/BasicSR")
import flow_visual
import matplotlib.pyplot as plt
import tensorboard
def visual_featuretensor(intensor):
    pass


def tensor2numpy(intensor):
    assert len(intensor.shape) == 4
    images = intensor.permute(0,2,3,1).cpu().numpy()
    return images
# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#### 医学超分临时可视化

def saveflow(flow):
    flow_np = flow[0].cpu().numpy()
    b,c,h,w = flow_np.shape
    for i in tqdm(range(0,b)):
        plt.figure()
        img = flow_visual.flow_to_image(flow_np[i:i+1,:,:,:])
        plt.imshow(img)
        plt.savefig("./msr_visual_result/flow/001/{}.png".format(str(i).zfill(4)))
        plt.close()

def savemsrimg(imgtensor):
    imgstensor = imgtensor[0]
    imgs_numpy = tensor2numpy(imgstensor)
    b,c,h,w = imgs_numpy.shape
    for i in range(0,b):
        plt.figure()
        img = normalization(imgs_numpy[i])
        plt.imshow(img)
        plt.savefig("./msr_visual_result/img/001/{}.png".format(str(i).zfill(4)))
        plt.close()