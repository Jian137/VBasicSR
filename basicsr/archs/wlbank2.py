from base64 import encode
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings



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
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256))
    def forward(self, x):

        y1s=[]# 输入7帧
        for i in x:
            b,c,h,w = i.size()
            q = i # 设定第4帧为当前帧

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

        return y1s
import torch
from torch.nn import functional as F
from torch import nn as nn

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x*1.0
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            assert 1==0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        res = self.conv(input)
        out = res + input
        return out


import numpy as np


class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,

                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=False,
                 ):
        super().__init__()

        ksz = 3

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 64,
            128: 64,
            256: 64,
            512: 32,
        }

        # 预设置
        max_depth=int(256//4//32) # int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))


        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                #ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2
        """
        if LQ_stage:
            # self.blocks.append(SwinLayers(**swin_opts))# RSTB 部分
            upsampler = nn.ModuleList()
            for i in range(2):
                in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
                upsampler.append(nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    )
                )
                res = res * 2


            self.blocks += upsampler

        self.LQ_stage = LQ_stage
        """
    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            # outputs.append(x)

        # return outputs
        return x
class VectorQuantizer(nn.Module):

    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight



        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q
class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input1):
        return self.block(input1)

class Decoder(nn.Module):
    def __init__(self,in_ch,out_ch,norm_type, act_type):
        super().__init__()
        self.decoder_group=[]# nn.Sequential()
        self.in_ch=in_ch
        self.out_ch = out_ch
        self.norm_type = norm_type
        self.act_type = act_type
        for _ in range(0,2):
            self.decoder_group.append(DecoderBlock(self.in_ch, self.out_ch, self.norm_type, self.act_type))  # type: ignore
        self.decoder = nn.Sequential(*self.decoder_group)
    def forward(self,x):
        return self.decoder(x)

class Bank(nn.Module):
    def __init__(self,feats_ch):
        super().__init__()
        self.feats_ch = feats_ch
        self.encode = Encoder(self.feats_ch,max_depth=2)
        self.vec = VectorQuantizer(feats_ch,feats_ch)
        self.decoder = Decoder(feats_ch,feats_ch,'gn','leakyrelu')
    def forward(self,x):
        out = self.encode(x)
        out = self.vec(out)
        out = self.decoder(out[0])
        return out
from thop import profile
if __name__=="__main__":

    encoder = Encoder(32,2)
    out = encoder(torch.rand(1,32,64,128))
    flops,params = profile(encoder,inputs=(torch.rand(1,32,128,128),))
    print(flops)
    print(params)
    print(encoder)
    bank = Bank(64)
    out = bank(torch.rand(3,64,128,64))
    print(out.shape)
    """
    vc = VectorQuantizer(64,64)

    dec = Decoder(64,64,'gn','leakyrelu')
    """