from dataclasses import dataclass
import torch
from torch import Tensor, nn
from einops import rearrange


@dataclass
class AutoEncoderParams:
    """
    自动编码器参数类，用于存储自动编码器的配置参数。

    属性:
        resolution (int): 输入图像的分辨率。
        in_channels (int): 输入图像的通道数。
        ch (int): 初始通道数，用于定义模型中各层的通道数。
        out_ch (int): 输出图像的通道数。
        ch_mult (List[int]): 通道数的倍数列表，用于定义每个分辨率阶段中通道数的增长。
        num_res_blocks (int): 每个分辨率阶段中残差块的数目。
        z_channels (int): 潜在空间（z 空间）的通道数。
        scale_factor (float): 缩放因子，用于调整潜在空间的大小。
        shift_factor (float): 平移因子，用于调整潜在空间的位置。
    """
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    """
    Swish 激活函数。

    Swish 是一种平滑且非线性的激活函数，定义为 x * sigmoid(x)。

    参数:
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 经过 Swish 激活函数处理后的张量。
    """
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """
    自注意力块（Attention Block）。

    该模块实现了自注意力机制，用于捕捉输入特征之间的全局依赖关系。
    """
    def __init__(self, in_channels: int):
        """
        初始化自注意力块。

        参数:
            in_channels (int): 输入的通道数。
        """
        super().__init__()
        self.in_channels = in_channels

        # 定义组归一化层
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # 定义查询（Q）、键（K）和值（V）的卷积层
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 查询卷积层
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 键卷积层
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 值卷积层

        # 定义输出投影卷积层
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """
        计算自注意力。

        参数:
            h_ (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过自注意力处理后的张量。
        """
        # 对输入进行归一化
        h_ = self.norm(h_)
        q = self.q(h_) # 计算查询
        k = self.k(h_) # 计算键
        v = self.v(h_) # 计算值

        # 获取批大小 (b)、通道数 (c)、高度 (h) 和宽度 (w)
        b, c, h, w = q.shape
        # 重塑张量，从 (b, c, h, w) 变为 (b, 1, h*w, c)
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # 计算缩放点积注意力
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 重塑张量，从 (b, 1, h*w, c) 变为 (b, c, h, w)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，执行自注意力块的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 自注意力块的输出。
        """
        # 添加残差连接
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    残差块（ResnetBlock）。

    该模块实现了残差连接和归一化、激活、卷积操作，用于构建深层神经网络。
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化残差块。

        参数:
            in_channels (int): 输入的通道数。
            out_channels (Optional[int], 可选): 输出的通道数。如果为 None，则输出通道数与输入通道数相同。
        """
        super().__init__()
        self.in_channels = in_channels
        # 如果未指定输出通道数，则输出通道数与输入通道数相同
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # 定义第一个归一化层（组归一化）
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 定义第二个归一化层（组归一化）
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 如果输入通道数与输出通道数不同，则定义一个1x1卷积层用于调整通道数
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播函数，执行残差块的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 残差块的输出。
        """
        # 保存输入张量
        h = x
        # 应用第一个归一化层
        h = self.norm1(h)
        # 应用 Swish 激活函数
        h = swish(h)
        # 应用第一个卷积层
        h = self.conv1(h)

        # 应用第二个归一化层
        h = self.norm2(h)
        # 应用 Swish 激活函数
        h = swish(h)
        # 应用第二个卷积层
        h = self.conv2(h)

        # 如果输入通道数与输出通道数不同，则应用1x1卷积层调整通道数
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        # 添加残差连接
        return x + h


class Downsample(nn.Module):
    """
    下采样模块（Downsample）。

    该模块通过卷积操作实现下采样功能，用于构建编码器部分。
    """
    def __init__(self, in_channels: int):
        """
        初始化下采样模块。

        参数:
            in_channels (int): 输入的通道数。
        """
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        # 在 PyTorch 的卷积中，没有非对称填充，必须手动实现
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        """
        前向传播函数，执行下采样操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 下采样后的张量。
        """
        # 定义填充参数，右边和底部填充1
        pad = (0, 1, 0, 1)
        # 对输入张量进行填充
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        # 应用卷积操作实现下采样
        x = self.conv(x)
        # 返回下采样后的张量
        return x


class Upsample(nn.Module):
    """
    上采样模块（Upsample）。

    该模块通过插值和卷积操作实现上采样功能，用于构建解码器部分。
    """
    def __init__(self, in_channels: int):
        """
        初始化上采样模块。

        参数:
            in_channels (int): 输入的通道数。
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        """
        前向传播函数，执行上采样操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 上采样后的张量。
        """
        # 使用最近邻插值进行上采样
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 应用卷积操作
        x = self.conv(x)
        # 返回上采样后的张量
        return x


class Encoder(nn.Module):
    """
    编码器（Encoder）。

    该模块实现了自动编码器中的编码器部分，用于将输入数据压缩到潜在空间。
    编码器包含多个下采样阶段，每个阶段包含多个残差块和可选的自注意力机制。
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        """
        初始化编码器。

        参数:
            resolution (int): 输入图像的分辨率。
            in_channels (int): 输入图像的通道数。
            ch (int): 初始通道数。
            ch_mult (List[int]): 通道数的倍数列表，用于定义每个分辨率阶段中通道数的增长。
            num_res_blocks (int): 每个分辨率阶段中残差块的数目。
            z_channels (int): 潜在空间（z 空间）的通道数。
        """
        super().__init__()
        # 初始通道数
        self.ch = ch
        # 分辨率阶段的数量
        self.num_resolutions = len(ch_mult)
        # 每个分辨率阶段中残差块的数目
        self.num_res_blocks = num_res_blocks
        # 输入图像的分辨率
        self.resolution = resolution
        # 输入图像的通道数
        self.in_channels = in_channels

        # downsampling
        # 下采样阶段
        # 输入卷积层
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # 当前分辨率
        curr_res = resolution
        # 输入通道数倍数列表
        in_ch_mult = (1,) + tuple(ch_mult)
        # 保存输入通道数倍数列表
        self.in_ch_mult = in_ch_mult
        # 定义下采样模块列表
        self.down = nn.ModuleList()

        # 残差块的输入通道数
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            # 定义当前分辨率阶段的残差块列表
            block = nn.ModuleList()
            # 定义当前分辨率阶段的自注意力模块列表
            attn = nn.ModuleList()
            # 计算当前阶段的输入通道数
            block_in = ch * in_ch_mult[i_level]
            # 计算当前阶段的输出通道数
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out

            # 定义当前下采样模块
            down = nn.Module()
            # 设置当前下采样模块的残差块
            down.block = block
            # 设置当前下采样模块的自注意力模块
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # 如果不是最后一个阶段，则添加下采样操作
                down.downsample = Downsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res // 2
            # 添加当前下采样模块到下采样模块列表
            self.down.append(down)

        # middle
        # 中间阶段
        # 定义中间模块
        self.mid = nn.Module()
        # 添加第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 添加自注意力模块
        self.mid.attn_1 = AttnBlock(block_in)
        # 添加第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        # 输出阶段
        # 输出归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 输出卷积层
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，执行编码器的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码器的输出。
        """
        # downsampling
        # 下采样阶段
        # 应用输入卷积层
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # 应用当前残差块
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    # 如果存在自注意力模块，则应用
                    h = self.down[i_level].attn[i_block](h)
                # 添加当前输出到列表中
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # 应用下采样操作
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # 中间阶段
        # 获取最后一个输出
        h = hs[-1]
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用自注意力模块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # end
        # 输出阶段
        # 应用归一化
        h = self.norm_out(h)
        # 应用 Swish 激活函数
        h = swish(h)
        # 应用输出卷积层
        h = self.conv_out(h)

        # 返回编码器的输出
        return h


class Decoder(nn.Module):
    """
    解码器（Decoder）。

    该模块实现了自动编码器中的解码器部分，用于将潜在空间的数据解码回原始数据空间。
    解码器包含多个上采样阶段，每个阶段包含多个残差块和可选的自注意力机制，用于逐步恢复输入数据的分辨率和细节。
    """
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        """
        初始化解码器。

        参数:
            ch (int): 初始通道数。
            out_ch (int): 输出图像的通道数。
            ch_mult (List[int]): 通道数的倍数列表，用于定义每个分辨率阶段中通道数的增长。
            num_res_blocks (int): 每个分辨率阶段中残差块的数目。
            in_channels (int): 输入潜在空间的通道数。
            resolution (int): 输出图像的分辨率。
            z_channels (int): 潜在空间（z 空间）的通道数。
        """
        super().__init__()
        # 初始通道数
        self.ch = ch
        # 分辨率阶段的数量
        self.num_resolutions = len(ch_mult)
        # 每个分辨率阶段中残差块的数目
        self.num_res_blocks = num_res_blocks
        # 输出图像的分辨率
        self.resolution = resolution
        # 输入潜在空间的通道数
        self.in_channels = in_channels
        # 计算缩放因子，用于确定最低分辨率
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        # 计算最低分辨率下的通道数、块输入通道数和当前分辨率
        # 计算最低分辨率下的块输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算最低分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 定义潜在空间的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        # 从潜在空间到块输入通道数的转换
        # 输入卷积层
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        # 中间阶段
        # 定义中间模块
        self.mid = nn.Module()
        # 添加第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 添加自注意力模块
        self.mid.attn_1 = AttnBlock(block_in)
        # 添加第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        # 上采样阶段
        # 定义上采样模块列表
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # 定义当前分辨率阶段的残差块列表
            block = nn.ModuleList()
            # 定义当前分辨率阶段的自注意力模块列表
            attn = nn.ModuleList()
            # 计算当前阶段的输出通道数
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out
            
            # 定义当前上采样模块
            up = nn.Module()
            # 设置当前上采样模块的残差块
            up.block = block
            # 设置当前上采样模块的自注意力模块
            up.attn = attn
            if i_level != 0:
                # 如果不是最后一个阶段，则添加上采样操作
                up.upsample = Upsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将当前上采样模块添加到列表的前面，以保持顺序一致
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        # 输出阶段
        # 输出归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 输出卷积层
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播函数，执行解码器的操作。

        参数:
            z (torch.Tensor): 输入的潜在空间张量。

        返回:
            torch.Tensor: 解码器的输出。
        """
        # 从潜在空间到块输入通道数的转换
        # z to block_in
        # 应用输入卷积层
        h = self.conv_in(z)

        # middle
        # 中间阶段
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用自注意力模块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # upsampling
        # 上采样阶段
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # 应用当前残差块
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    # 如果存在自注意力模块，则应用
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 应用上采样操作
                h = self.up[i_level].upsample(h)

        # end
        # 输出阶段
        # 应用归一化
        h = self.norm_out(h)
        # 应用 Swish 激活函数
        h = swish(h)
        # 应用输出卷积层
        h = self.conv_out(h)

        # 返回解码器的输出
        return h


class DiagonalGaussian(nn.Module):
    """
    对角高斯分布（DiagonalGaussian）。

    该模块实现了对角高斯分布，用于在变分自编码器（VAE）中采样潜在变量。
    """
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        """
        初始化对角高斯分布。

        参数:
            sample (bool, 可选): 是否进行采样。如果为 False，则只返回均值。
            chunk_dim (int, 可选): 分割维度的索引，用于将输入张量分割为均值和对数方差。
        """
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播函数，执行对角高斯分布的操作。

        参数:
            z (torch.Tensor): 输入张量，包含均值和对数方差。

        返回:
            torch.Tensor: 采样后的潜在变量或均值。
        """
        # 将输入张量分割为均值和对数方差
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            # 计算标准差
            std = torch.exp(0.5 * logvar)
            # 从标准正态分布中采样，并与均值和标准差结合
            return mean + std * torch.randn_like(mean)
        else:
            # 如果不采样，则只返回均值
            return mean


class AutoEncoder(nn.Module):
    """
    自动编码器（AutoEncoder）。

    该模块实现了基于对角高斯分布的变分自编码器（VAE），用于将输入数据编码到潜在空间并解码回原始数据空间。
    """
    def __init__(self, params: AutoEncoderParams):
        """
        初始化自动编码器。

        参数:
            params (AutoEncoderParams): 自动编码器的参数，包括分辨率、通道数、通道倍数、残差块数等。
        """
        super().__init__()

        # 初始化编码器
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        # 初始化解码器
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        # 初始化对角高斯分布模块
        self.reg = DiagonalGaussian()

        # 设置缩放因子和平移因子
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        """
        编码函数，将输入数据编码到潜在空间。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 编码后的潜在变量。
        """
        # 对输入数据进行编码，并应用对角高斯分布
        z = self.reg(self.encoder(x))
        # 对潜在变量进行缩放和平移
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        解码函数，将潜在变量解码回原始数据空间。

        参数:
            z (torch.Tensor): 潜在变量。

        返回:
            torch.Tensor: 解码后的数据。
        """
        # 对潜在变量进行逆缩放和平移
        z = z / self.scale_factor + self.shift_factor
        # 对潜在变量进行解码
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，执行自动编码器的操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 自动编码器的输出。
        """
        # 编码和解码输入数据
        return self.decode(self.encode(x))
