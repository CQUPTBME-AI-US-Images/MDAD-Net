import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class ADatt(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, ad_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  #特征通道维度
        self.num_patches = num_patches  #输入序列的长度
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))  #将序列长度还原为二维特征图的尺寸（num_patches=49则window_size=(7,7)）
        self.window_size = window_size  #特征图尺寸
        self.num_heads = num_heads  #多头注意力的头数
        head_dim = dim // num_heads  #单头维度
        self.scale = head_dim ** -0.5  #注意力缩放因子

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  #线性层，将输入特征映射为Q向量，输入输出维度都是dim
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  #线性层，同时映射K和V，输出维度dim*2，后续拆分
        self.attn_drop = nn.Dropout(attn_drop)  #对注意力权重dropout
        self.proj = nn.Linear(dim, dim)  #输出的投影层
        self.proj_drop = nn.Dropout(proj_drop)  #对最终投影结果应用 dropout

        self.sr_ratio = sr_ratio  #空间压缩比，>1时对K和V降采样
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  #卷积层
            self.norm = nn.LayerNorm(dim)  #LN层

        self.ad_num = ad_num  #自适应token数量
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)  #DWConv
        #6个用于不同注意力方向的可学习偏置
        self.an_bias = nn.Parameter(torch.zeros(num_heads, ad_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, ad_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, ad_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, ad_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, ad_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], ad_num))
        #用trunc_normal_初始化6个位置偏置参数，标准差 0.02
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(ad_num ** 0.5)  #自适应池化的输出尺寸，ad_num=49则pool_size=7
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))  #自适应平均池化层，生成自适应token
        self.softmax = nn.Softmax(dim=-1)  #注意力权重归一化

    def forward(self, x, H, W):
        b, n, c = x.shape  #解包x的形状，保存批量大小b、token数n、通道c（n=H*W）
        num_heads = self.num_heads
        head_dim = c // num_heads  #取出多头数num_heads和单头维度head_dim
        q = self.q(x)  #将输入x投影为Q向量，形状(b, n, c)
        #对K和V降采样
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W) #将x从(b,n,c)转为(b,c,n)并还原为图像格式(b,c,H,W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1) #卷积降采样，特征图尺寸变为H/sr_ratio*W/sr_ratio，重新展平为序列格式(b, n', c)
            x_ = self.norm(x_)  #对降采样后的特征做LayerNorm
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)  #输出形状(b, n', 2c),拆分为K/V维度，形状(b, n', 2, c)，维度重排为(2, b, n', c)
        k, v = kv[0], kv[1] #分离K和V,形状均为(b, n', c)

        ad_trans = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)  #将Q从(b,n,c)转为图像格式(b,c,H,W)，自适应池化输出(b,c,pool_size,pool_size)展平为序列格式(b, ad_num, c)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3) #将Q从(b,n,c)拆分为多头，形状(b, n, num_heads, head_dim),维度重排为(b, num_heads, n, head_dim)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3) #b, nh, kv_n, hd
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3) #b, nh, kv_n, hd
        ad_trans = ad_trans.reshape(b, self.ad_num, num_heads, head_dim).permute(0, 2, 1, 3) #重塑为多头格式，形状(b, num_heads, ad_num, head_dim)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)  #K/V降采样后的特征图尺寸
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.ad_num, -1).repeat(b, 1, 1, 1)  #对an_bias双线性插值匹配kv_size，然后reshape为(1, num_heads, ad_num, n')，并repeat批量维度b
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.ad_num, -1).repeat(b, 1, 1, 1) #ah_bias+aw_bias（高度+宽度偏置），同样reshape+repeat
        position_bias = position_bias1 + position_bias2 #总位置偏置
        ad_attn = self.softmax((ad_trans * self.scale) @ k.transpose(-2, -1) + position_bias) #注意力权重ad_attn=自适应token乘缩放因子的结果使用点积计算相似度得到(b, nh, ad_num, n'),添加位置偏置后归一化
        ad_attn = self.attn_drop(ad_attn)  #对注意力权重应用dropout
        ad_v = ad_attn @ v #注意力加权求和，V是(b, nh, n', hd)，输出结果为(b, nh, ad_num, hd)）

        ad_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        ad_bias1 = ad_bias1.reshape(1, num_heads, self.ad_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)  #对na_bias插值到window_size，维度重排为(1, nh, n, ad_num)
        ad_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.ad_num).repeat(b, 1, 1, 1)  #ha_bias + wa_bias，形状(1, nh, n, ad_num)
        ad_bias = ad_bias1 + ad_bias2  #总偏置
        q_attn = self.softmax((q * self.scale) @ ad_trans.transpose(-2, -1) + ad_bias)
        q_attn = self.attn_drop(q_attn)  #计算Q对自适应token的注意力权重q_attn
        x = q_attn @ ad_v #注意力加权求和，形状(b, nh, n, hd)

        x = x.transpose(1, 2).reshape(b, n, c)  #将x从(b, nh, n, hd)还原为(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)  #将V从多头格式还原为图像格式(b, H/sr_ratio, W/sr_ratio, c)，再permute为(b, c, H/sr_ratio, W/sr_ratio)
        if self.sr_ratio > 1:  #若sr_ratio>1，双线性插值将V上采样回原尺寸(H,W)
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c) #对V做深度卷积，将卷积后的V转回序列格式(b,n,c)，并通过残差链接融合注意力输出和深度卷积的V特征

        x = self.proj(x)  #线性投影
        x = self.proj_drop(x)  # dropout
        return x  #b，n，c



if __name__ == '__main__':
    dim = 64
    num_patches = 49

    block = ADatt(dim=dim, num_patches=num_patches)

    H, W = 7, 7
    x = torch.rand(1, num_patches, dim)

    # Forward pass
    output = block(x, H, W)
    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")
