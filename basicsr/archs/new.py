from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY

class MyModelV1(nn.Module):
    def __init__(self,mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 cpu_cache_length=100):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ConvResidualBlocks(mid_channels, mid_channels, 5))
            #TODO npy输入为1通道
        self.feat_MMA1=MMA(self.mid_channels)# 可替换为类似GLEAN类网络
    def forward(self,lqs):
        n,t,c,h,w=lqs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False

        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))# 提取浅层特征
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
        #---------------------浅层特征提取------------------------------------------------
        self.feat_Q = self.feat_MMA1(feats['spatial'])# 充当流的功能


import torch.nn.functional as F
class MMA(nn.Module):
    # CrossFrame Non-Local Attention
    def __init__(self, channels):
        super(MMA, self).__init__()
        self.in_channels = channels # config['in_channels']
        self.inter_channels = self.in_channels # // 2

         # config['height']


        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)

        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z1.weight, 0)
        nn.init.constant_(self.W_z1.bias, 0)
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 512))
    def forward(self, x):

        y1s=[]# 输入7帧
        n,t,c,h,w = x.size()
        for i in range(0,t):

            q = i[:,i,:,:,:] # 设定第4帧为当前帧

            q_=self.norm(q)  #F_{t}

            phi_x = self.phi(q_).view(b,self.inter_channels, -1) # 生成Q


            phi_x_for_quant=phi_x.permute(0,2,1)
            phi_x= phi_x.permute(0,2,1).contiguous()# 意义不明？



            mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
            f1 = torch.matmul(phi_x_for_quant, mbg)# 从Q中计算Y_t
            f_div_C1 = F.softmax(f1 * (int(self.inter_channels) ** (-0.5)), dim=-1)
            y1 = torch.matmul(f_div_C1, mbg.permute(0, 2, 1))
            qloss=torch.mean(torch.abs(phi_x_for_quant-y1))
            y1 = y1.permute(0, 2, 1).view(b, self.inter_channels, h, w).contiguous()
            W_y1 = self.W_z1(y1)
            y1s.append(W_y1+i)

        return torch.stack(y1s,dim=1)