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


@ARCH_REGISTRY.register()
class BasicVSRSubSubX4(nn.Module):
    # 无光流模块的basicvsr++
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow

        #self.mma=MMA(mid_channels)

        self.catconv = nn.Conv2d(2*mid_channels,mid_channels,3,1,1)
        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(1, mid_channels, 5)#TODO npy输入为1通道
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ConvResidualBlocks(mid_channels, mid_channels, 5))
            #TODO npy输入为1通道
        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)#TODO npy输出为1通道
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True



    def propagate(self, feats, module_name):
        # 修改，输入不包括flows
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        #n, t, _, h, w = flows.size()
        t = len(feats['spatial'])
        n, _, h, w = feats['spatial'][0].size()
        # frame_idx = range(0, t + 1)
        frame_idx = range(0, t)
        #flow_idx = range(-1, t)# 删除
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            #flow_idx = frame_idx# 删除

        # feat_prop = flows.new_zeros(n, self.mid_channels, h, w)# 修改
        feat_prop = feats['spatial'][0].new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                #flow_n1 = flows[:, flow_idx[i], :, :, :]# 删除
                #if self.cpu_cache:
                #    flow_n1 = flow_n1.cuda()# 删除

                #cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))# 修改融合方式或删除
                #  cond_n1 = feat_prop
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                #flow_n2 = torch.zeros_like(flow_n1)# 删除
                # cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    #flow_n2 = flows[:, flow_idx[i - 1], :, :, :]# 删除
                    #if self.cpu_cache:
                    #    flow_n2 = flow_n2.cuda()# 删除

                    #flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))# 修改融合方式
                    #cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    # cond_n2 = feat_n2
                # flow-guided deformable convolution
                        # 全零矩阵  当前帧     二阶传播特征
                #cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)# 加入卷积
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.catconv(feat_prop)
                #feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)# 删除可变形对齐 改成简单卷积

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)# 确定大小 128通道
            feat_prop = feat_prop + self.backbone[module_name](feat)# 修改

            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        #feats['spatial'] = self.mma(feats['spatial'])
        # 添加记忆模块
        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
        #浅层特征提取
        # compute optical flow using the low-res inputs
        # assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
        #    'The height and width of low-res inputs must be at least 64, '
        #    f'but got {h} and {w}.')
        # flows_forward, flows_backward = self.compute_flow(lqs_downsample)# 删除
        # 计算光流
        # feature propgation
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
                if self.cpu_cache:
                    #del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)


# if __name__ == '__main__':
#     spynet_path = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
#     model = BasicVSRPlusPlus(spynet_path=spynet_path).cuda()
#     input = torch.rand(1, 2, 3, 64, 64).cuda()
#     output = model(input)
#     print('===================')
#     print(output.shape)
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


from thop import profile
import torch
if __name__=="__main__":
    """
    encoder = Encoder(32,2)
    flops,params = profile(encoder,inputs=(torch.rand(1,32,128,128),))
    print(flops)
    print(params)
    print(encoder)
    """
    model=BasicVSRPlusPlusMMAX2()
    inputs = torch.rand(1,7,1,256,256)
    print(model(inputs).shape)