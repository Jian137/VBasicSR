import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AC(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., kmeans = False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
    def forward(self,x,x_size,mask=None):
        H,W = x_size
        B_,N,C=x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q_pre = qkv[0].reshape(B_*self.num_heads,N, C // self.num_heads).permute(0,2,1)

        ntimes = int(math.log(N//(self.ws*self.ws),2))

        q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_*self.num_heads,N) #3,3136 # 序列号？
        if False:
            for i in range(ntimes):
                bh,d,n=q_pre.shape
                q_pre_new = q_pre.reshape(bh,d,2,n//2)
                q_avg = q_pre_new.mean(dim=-1)
                q_avg = nn.functional.normalize(q_avg,dim=-2)
                    # 0 3,32,2
                iters = 2

                for i in range(iters):
                    q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1),dim=-1).bmm(q_avg) # 3,3136,2
                    soft_assign = torch.nn.functional.softmax(q_scores*100, dim=-1).detach()
                    q_avg = q_pre.bmm(soft_assign)
                    q_avg = torch.nn.functional.normalize(q_avg,dim=-2)
                q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1),dim=-1).bmm(q_avg).reshape(bh,n,2)#.unsqueeze(2)
                q_idx = (q_scores[:,:,0]+1)/(q_scores[:,:,1]+1)
                _,q_idx = torch.sort(q_idx,dim=-1)
                q_idx_last = q_idx_last.gather(dim=-1,index=q_idx).reshape(bh*2,n//2)
                q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
                q_pre = q_pre.gather(dim=-1,index=q_idx).reshape(bh,d,2,n//2).permute(0,2,1,3).reshape(bh*2,d,n//2)

        if True:
            bh,d,n=q_pre.shape
            q_pre_new = q_pre.reshape(bh,d,n//self.ws**2,self.ws**2)
            # 每组 self.ws**2
            q_avg = q_pre_new.mean(dim=-1)
            q_avg = nn.functional.normalize(q_avg,dim=-2)

            iters = 2
            for i in range(iters):
                q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1),dim=-1).bmm(q_avg) # 3,3136,2
                soft_assign = torch.nn.functional.softmax(q_scores*100, dim=-1).detach()
                q_avg = q_pre.bmm(soft_assign)
                q_avg = torch.nn.functional.normalize(q_avg,dim=-2)
            q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1),dim=-1).bmm(q_avg).reshape(bh,n,n//self.ws**2)#.unsqueeze(2)
            q_idx = (q_scores[:,:,0]+1)/(q_scores[:,:,1]+1)
            _,q_idx = torch.sort(q_idx,dim=-1)
            q_idx_last = q_idx_last.gather(dim=-1,index=q_idx).reshape(bh*n//self.ws**2,self.ws**2)
            q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
            q_pre = q_pre.gather(dim=-1,index=q_idx).reshape(bh,d,n//self.ws**2,self.ws**2).permute(0,2,1,3).reshape(bh*n//self.ws**2,d,self.ws**2)


        q_idx = q_idx_last.view(B_,self.num_heads,N)
        _,q_idx_rev = torch.sort(q_idx,dim=-1)
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
        qkv_pre = qkv.gather(dim=-2,index=q_idx)


        #q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=self.ws*self.ws)

        q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) ws (h c)', ws=self.ws*self.ws)

        attn = (q @ k.transpose(-2, -1))

        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0) # TODO
        """

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)



        out = rearrange(out,'(b nw)  ws (h d) -> b (h d) nw ws', h=self.num_heads, b=B_)

        v = rearrange(v,'(b nw) ws (h c) -> (b nw) h ws c',h=self.num_heads,b=B_)

        # TODO V 整形
        v = rearrange(v[:,:,:self.ws*self.ws,:], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
        #W = int(math.sqrt(N)) # windows_size

        out = out.reshape(B_,self.num_heads,C//self.num_heads,-1)
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        x = out.gather(dim=-1,index=q_idx_rev).reshape(B_,C,N).permute(0,2,1)
        v = v.gather(dim=-1,index=q_idx_rev).reshape(B_,C,H,W)
        v = self.get_v(v)
        v = v.reshape(B_,C,N).permute(0,2,1)
        x = x + v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__=="__main__":
    model = AC(60,8,3).cuda()
    inputs=torch.randn(1,4096,60).cuda()
    print(model(inputs).shape)