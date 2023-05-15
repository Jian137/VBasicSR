import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile

from basicsr.archs.arch_util import ResidualBlockNoBN,make_layer
from basicsr.utils.registry import ARCH_REGISTRY
class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)

        return fx + x

# 深蓝色方块
class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
        )

    def forward(self, x):
        return self.f(x)

# 橙色方块
class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out)
        )

    def forward(self, x):
        return self.f(x)

class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs = [32, 64, 96], nrow=3, ncol=6):
        super(GridNet, self).__init__()

        self.n_row = nrow
        self.n_col = ncol
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col-1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col/2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col/2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        forward_func = getattr(self, f'forward_{self.n_row}{self.n_col}')
        return forward_func(x)

    def forward_36(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)  # type: ignore
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)

    def forward_34(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_22 = self.lateral_2_1(state_21)
        state_12 = self.up_1_0(state_22) + self.lateral_1_1(state_11)
        state_02 = self.up_0_0(state_12) + self.lateral_0_1(state_01)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_1(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_1(state_13) + self.lateral_0_2(state_02)

        return self.lateral_final(state_03)

    def forward_32(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_21 = self.lateral_2_0(state_20)
        state_11 = self.up_1_0(state_21) + self.lateral_1_0(state_10)
        state_01 = self.up_0_0(state_11) + self.lateral_0_0(state_00)

        return self.lateral_final(state_01)

    def forward_30(self, x):
        state_00 = self.lateral_init(x)

        return self.lateral_final(state_00)

    def forward_24(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)

        state_12 = self.lateral_1_1(state_11)
        state_02 = self.up_0_0(state_12) + self.lateral_0_1(state_01)

        state_13 = self.lateral_1_2(state_12)
        state_03 = self.up_0_1(state_13) + self.lateral_0_2(state_02)

        return self.lateral_final(state_03)

    def forward_22(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)

        state_11 = self.lateral_1_0(state_10)
        state_01 = self.up_0_0(state_11) + self.lateral_0_0(state_00)

        return self.lateral_final(state_01)

class down(nn.Module):
    def __init__(self, num_in_ch, num_out_ch):
        super(down, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, stride=2, padding=1),
            nn.PReLU(num_out_ch),
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch),
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )

    def forward(self, x):
        x = self.body(x)

        return x

class up(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, split=0.5):
        super(up, self).__init__()
        self.split = split
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_in_ch,  num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )
        self.decouple = nn.Sequential(
            nn.Conv2d(2*num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
            )

    def forward(self, x, skip_ch, t):
        x = self.head(x)
        x = self.decouple(torch.cat((x, skip_ch), 1))
        b, c, h, w = x.shape
        p = int(c*self.split)
        x = torch.cat((x[:,:p]*t, x[:,p:]), 1)
        x = self.merge(x)

        return x


class SDLNet(nn.Module):
    """
    SDLNet architecture
    """

    def __init__(self, num_in_ch, num_out_ch, split=0.5, num_feat=32, nrow=3, ncol=6):
        super(SDLNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, 32, 3, stride=1, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.PReLU(32)
        )
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)

        self.up1   = up(512, 512, split)
        self.up2   = up(512, 256, split)
        self.up3   = up(256, 128, split)
        self.up4   = up(128, 64, split)
        self.up5   = up(64, 32, split)
        self.tail = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # 32
            nn.PReLU(32)
        )

        self.gridnet = GridNet(32, num_out_ch, nrow=nrow, ncol=ncol)

    def preforward(self, x, t):
        t = t.view(-1, 1, 1, 1)

        s1 = self.head(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)

        x  = self.up1(x, s5, t)
        x  = self.up2(x, s4, t)
        x  = self.up3(x, s3, t)
        x  = self.up4(x, s2, t)
        x  = self.up5(x, s1, t)
        x  = self.tail(x)

        return x

    def forward(self, x, t):
        x_01 = self.preforward(x, t)
        '''
        idx_rvs = torch.LongTensor(range(-3, 3))
        x_rvs = x[:,idx_rvs]
        x_10 = self.preforward(x_rvs, 1-t)

        x = self.gridnet(torch.cat((x_01, x_10), 1))
        '''
        x = self.gridnet(x_01)

        return x_01

class up2(nn.Module):
    def __init__(self, num_in_ch, num_out_ch):
        super(up2, self).__init__()

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_in_ch,  num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )
        self.decouple = nn.Sequential(
            nn.Conv2d(2*num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )


        self.merge = nn.Sequential(
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
            )


    def forward(self, x, skip_ch):
        x = self.head(x)
        x = self.decouple(torch.cat((x, skip_ch), 1))
        b, c, h, w = x.shape
        #p = int(c*self.split)
        #x = torch.cat((x[:,:p], x[:,p:]), 1)
        x = self.merge(x)

        return x
class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
class latentcbb(nn.Module):
    def __init__(self,mid_channels=512,num_blocks=7):
        super().__init__()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        self.catconv = nn.Conv2d(mid_channels//4,mid_channels//8,3,1,1)
        self.backbone = nn.ModuleDict()
        for i,module in enumerate(modules):
            self.backbone[module] = ConvResidualBlocks((8 + i) * mid_channels//8, mid_channels//8, num_blocks)
        self.mid_channels = mid_channels
        #self.conv2 = nn.Conv2d(mid_channels,mid_channels,3,1,1)
    def propagate(self,feats,module_name):
        # 输入改成字典形式{'spatial':[],module_name:[]}
        t= len(feats['spatial'])
        n,_,h,w = feats['spatial'][0].size()
        n = 1
        frame_idx = range(0, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
        feat_prop = feats['spatial'][0].new_zeros(n, self.mid_channels//8, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            """if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment"""
            c = self.mid_channels-self.mid_channels//8
            current_p = feat_current[:,:c,:,:]
            current_q = feat_current[:,c:,:,:]


            if i > 0 :
                feat_n2 = current_q #torch.zeros_like(feat_prop)
                if i > 1:
                    feat_n2 = feats[module_name][-2]# 二阶部分
                    """if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()"""
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)# 与前两帧信息聚合
                feat_prop = self.catconv(feat_prop)
        # concatenate and residual blocks
            feat = [current_p] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            # 将p与传播中其他帧信息融合
            """if self.cpu_cache:
                feat = [f.cuda() for f in feat]"""

            feat = torch.cat(feat, dim=1)# 确定大小


            feat_prop = feat_prop + self.backbone[module_name](feat)# 修改 输出通道与feat_poro一样
            feats[module_name].append(feat_prop)


            feats['spatial'][mapping_idx[idx]] = torch.cat([current_p,feat_prop],dim=1) # TODO 通道数不对
            """if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()
            """
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        # 添加记忆模块
        return feats
    def forward(self,lqs):

        t, c, h, w = lqs.size()
        lqs = lqs.view(-1,t,c,h,w)
        feats = {}
        feats['spatial']=[]
        for i in range(0,t):
            feats['spatial'].append(lqs[:,i,:,:,:])
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    pass
                    # flows = flows_backward
                    """
                    elif flows_forward is not None:
                    flows = flows_forward
                    """
                else:
                    pass

                # feats = self.propagate(feats, flows, module)# 修改融合模块
                feats = self.propagate(feats, module)

        return feats
import torch.nn.functional as F
@ARCH_REGISTRY.register()
class Model(nn.Module):
    def __init__(self,num_in_ch,num_out_ch,scale=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, 32, 3, stride=1, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.PReLU(32)
        )
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1   = up2(512, 512 )
        self.up2   = up2(512, 256 )
        self.up3   = up2(256, 128 )
        self.up4   = up2(128, 64 )
        self.up5   = up2(64, 32 )
        self.tail = nn.Sequential(
            nn.Conv2d(32,num_out_ch*scale**2 , 3, stride=1, padding=1),# 32

            nn.PReLU(num_out_ch*scale**2),
            nn.PixelShuffle(scale)

        )
        self.latentcbb=latentcbb(512)
        self.scale = scale
    def forward(self,x):
        b,t,c,h,w = x.shape
        x1=x.view(-1,c,h,w)

        s1 = self.head(x1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        x2 = self.latentcbb(x)
        x2 = torch.cat(x2['spatial'])
        x  = self.up1(x2, s5 )
        x  = self.up2(x, s4 )
        x  = self.up3(x, s3 )
        x  = self.up4(x, s2 )
        x  = self.up5(x, s1 )
        x  = self.tail(x)
        x = F.interpolate(x1,scale_factor=self.scale,mode='bicubic') + x
        x = x.view(b,t,-1,h*self.scale ,w*self.scale)
        return x
if __name__=="__main__":
    model = Model(3,3)
    inputs = torch.rand(1,31,3,224,224)
    print(model(inputs).shape)
    a,b = profile(model,inputs=(inputs,))
    print(a)
    print(b)

    # model2 = latentcbb(512)
    # inputs = torch.rand(3,512,8,8)
    #print(model2(inputs)['spatial'].shape)
    # a,b = profile(model2,inputs=(inputs,))
    #print(a)
    #print(b)
"""
if __name__=="__main__":
    model = SDLNet(3,3)
    inputs = torch.rand(2,3,256,256)
    print(model(inputs,torch.tensor([[[[0.5]]]])).shape)
    a,b = profile(model,inputs=(inputs,torch.tensor([[[[0.5]]]]),))
    print(a)
    print(b)
"""