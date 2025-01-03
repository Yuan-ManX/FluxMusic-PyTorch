import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    实现多头自注意力机制，并应用旋转位置编码（RoPE）。

    参数:
        q (torch.Tensor): 查询张量，形状为 (B, H, L, D)。
        k (torch.Tensor): 键张量，形状为 (B, H, L, D)。
        v (torch.Tensor): 值张量，形状为 (B, H, L, D)。
        pe (torch.Tensor): 位置编码张量，形状为 (B, L, D)。

    返回:
        torch.Tensor: 注意力机制的输出，形状为 (B, L, H*D)。
    """
    # 应用旋转位置编码（RoPE）到查询和键
    q, k = apply_rope(q, k, pe)
    
    # 计算缩放点积注意力
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # 重塑输出张量，从 (B, H, L, D) 变为 (B, L, H*D)
    x = rearrange(x, "B H L D -> B L (H D)")
    
    # 返回注意力机制的输出
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    生成旋转位置编码（RoPE）。

    参数:
        pos (torch.Tensor): 输入的位置张量，形状为 (..., N)。
        dim (int): 位置编码的维度。
        theta (int): 位置编码的参数 θ，用于控制频率。

    返回:
        torch.Tensor: 旋转位置编码张量，形状为 (..., N, 2, 2)。
    """
    assert dim % 2 == 0
    # 计算缩放因子
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    # 计算角频率
    omega = 1.0 / (theta**scale)
    # 计算正弦和余弦项
    out = torch.einsum("...n,d->...nd", pos, omega)
    # 堆叠正弦和余弦项，形成 (..., N, 2, 2) 的张量
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    # 重塑张量
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    # 返回旋转位置编码张量
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    应用旋转位置编码（RoPE）到查询和键。

    参数:
        xq (torch.Tensor): 查询张量，形状为 (B, H, L, D)。
        xk (torch.Tensor): 键张量，形状为 (B, H, L, D)。
        freqs_cis (torch.Tensor): 旋转位置编码张量，形状为 (B, L, D, 2, 2)。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 应用 RoPE 后的查询和键张量。
    """
    # 重塑查询和键张量，以便应用 RoPE
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    # 应用 RoPE
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # 重塑回原始形状
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """
    多维嵌入层，用于生成旋转位置编码（RoPE）。

    该嵌入层生成的位置编码适用于多维输入，例如在多轴位置编码中使用。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        """
        初始化 EmbedND 嵌入层。

        参数:
            dim (int): 嵌入维度。
            theta (int): 位置编码的参数 θ，用于控制频率。
            axes_dim (list[int]): 每个轴的维度列表。
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播函数，生成旋转位置编码。

        参数:
            ids (torch.Tensor): 输入的 ID 张量，形状为 (..., N)。

        返回:
            torch.Tensor: 旋转位置编码张量，形状为 (..., N, D, 2, 2)。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        # 生成每个轴的旋转位置编码，并沿着最后一个轴连接起来
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        # 在倒数第二个轴上添加一个维度
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    生成时间步长的正弦位置嵌入。

    参数:
        t (torch.Tensor): 一个 1-D 张量，包含 N 个索引，每个批次元素一个。
                          这些索引可以是分数。
        dim (int): 输出的维度。
        max_period (int, 可选): 控制嵌入的最小频率，默认为 10000。
        time_factor (float, 可选): 时间步长的缩放因子，默认为 1000.0。

    返回:
        torch.Tensor: 位置嵌入张量，形状为 (N, D)。
    """
    # 缩放时间步长
    t = time_factor * t
    # 计算一半的维度
    half = dim // 2
    # 计算频率
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )
    
    # 计算角度
    args = t[:, None].float() * freqs[None]
    # 生成正弦和余弦嵌入
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # 如果维度是奇数，添加零填充
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        # 确保嵌入与输入张量类型一致
        embedding = embedding.to(t)

    # 返回位置嵌入张量
    return embedding


class MLPEmbedder(nn.Module):
    """
    多层感知机嵌入器（MLP Embedder）。

    该类通过一个线性层、一个 SiLU 激活函数以及另一个线性层，将输入嵌入到高维空间中。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        """
        初始化 MLPEmbedder。

        参数:
            in_dim (int): 输入的维度。
            hidden_dim (int): 隐藏层的维度。
        """
        super().__init__()
        # 第一个线性层，将输入维度映射到隐藏维度
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        # SiLU 激活函数
        self.silu = nn.SiLU()
        # 第二个线性层，保持维度不变
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数，执行嵌入操作。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 嵌入后的张量。
        """
        # 线性层 -> SiLU -> 线性层
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """
    均方根归一化层（Root Mean Square Normalization Layer）。

    该层对输入张量进行归一化处理，通过计算均方根值（RMS）来调整每个样本的尺度。
    """
    def __init__(self, dim: int):
        """
        初始化 RMSNorm。

        参数:
            dim (int): 输入张量的最后一个维度。
        """
        super().__init__()
        # 初始化缩放参数，可学习
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        """
        前向传播函数，执行 RMS 归一化。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 保存输入张量的数据类型
        x_dtype = x.dtype
        # 将输入张量转换为 float 类型
        x = x.float()
        # 计算均方根值（RMS），并加上一个小常数以避免除零
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # 应用归一化并乘以缩放参数
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """
    查询键归一化层（QK Normalization Layer）。

    该层分别对查询（Q）和键（K）进行归一化处理，以确保注意力机制的有效性。
    """
    def __init__(self, dim: int):
        """
        初始化 QKNorm。

        参数:
            dim (int): 输入张量的最后一个维度。
        """
        super().__init__()
        # 查询的 RMS 归一化
        self.query_norm = RMSNorm(dim)
        # 键的 RMS 归一化
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播函数，执行查询和键的归一化。

        参数:
            q (torch.Tensor): 查询张量。
            k (torch.Tensor): 键张量。
            v (torch.Tensor): 值张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 归一化后的查询和键张量。
        """
        # 对查询进行归一化
        q = self.query_norm(q)
        # 对键进行归一化
        k = self.key_norm(k)
        # 返回归一化后的查询和键张量，并转换数据类型
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """
    自注意力机制（Self-Attention）模块。

    该模块实现了多头自注意力机制，并应用查询键归一化（QK Normalization）。
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        """
        初始化 SelfAttention。

        参数:
            dim (int): 输入和输出的维度。
            num_heads (int, 可选): 多头注意力的头数，默认为 8。
            qkv_bias (bool, 可选): 在查询、键和值线性层中是否使用偏置，默认为 False。
        """
        super().__init__()
        # 多头注意力的头数
        self.num_heads = num_heads
        # 每个头的维度
        head_dim = dim // num_heads

        # 查询、键和值的线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 查询键归一化层
        self.norm = QKNorm(head_dim)
        # 输出投影线性层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播函数，执行自注意力机制。

        参数:
            x (torch.Tensor): 输入张量。
            pe (torch.Tensor): 位置编码张量。

        返回:
            torch.Tensor: 自注意力机制的输出。
        """
        # 计算查询、键和值
        qkv = self.qkv(x)
        # 重塑张量
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 对查询和键进行归一化
        q, k = self.norm(q, k, v)
        # 应用注意力机制
        x = attention(q, k, v, pe=pe)
        # 输出投影
        x = self.proj(x)
        # 返回自注意力机制的输出
        return x


class FlashSelfMHAModified(nn.Module):
    """
    self-attention with flashattention
    """
    """
    使用 FlashAttention 的自注意力机制。

    该模块实现了改进的 Flash 自注意力机制，通过 FlashAttention 加速多头自注意力计算。
    """
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_norm=True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=RMSNorm,
                 ):
        """
        初始化 FlashSelfMHAModified。

        参数:
            dim (int): 输入和输出的维度。
            num_heads (int): 多头注意力的头数。
            qkv_bias (bool, 可选): 在查询、键和值线性层中是否使用偏置，默认为 False。
            qk_norm (bool, 可选): 是否对查询和键进行归一化，默认为 True。
            attn_drop (float, 可选): 注意力 Dropout 概率，默认为 0.0。
            proj_drop (float, 可选): 输出投影 Dropout 概率，默认为 0.0。
            device (torch.device, 可选): 设备类型。
            dtype (torch.dtype, 可选): 数据类型。
            norm_layer (nn.Module, 可选): 归一化层类型，默认为 RMSNorm。
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # 输入和输出的维度
        self.dim = dim
        # 多头注意力的头数
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"

        # 每个头的维度
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        # 线性层，用于计算查询、键和值
        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: 如果使用 fp16，eps 应该设为 1 / 65530
        # 查询的归一化层，如果需要的话
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # 键的归一化层，如果需要的话
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # 输出投影线性层
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        # 输出投影 Dropout 层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pe):
        """
        前向传播函数，执行自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch, seqlen, hidden_dim)。
            pe (torch.Tensor): 位置编码张量，形状与 x 相同。

        返回:
            torch.Tensor: 自注意力机制的输出，形状为 (batch, seqlen, hidden_dim)。
        """
        # 获取批量大小 (b)、序列长度 (s) 和维度 (d)
        b, s, d = x.shape

        # 计算查询、键和值
        qkv = self.Wqkv(x)
        # 重塑张量为 [b, s, 3, h, d]
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        # 解绑为 [b, s, h, d]
        q, k, v = qkv.unbind(dim=2) # [b, s, h, d]

        # 对查询进行归一化，并转换为半精度
        q = self.q_norm(q).half()   # [b, s, h, d]
        # 对键进行归一化，并转换为半精度
        k = self.k_norm(k).half()
        # 应用旋转位置编码（RoPE）
        q, k = apply_rope(q, k, pe)

        # 堆叠为 [b, s, 3, h, d]
        qkv = torch.stack([q, k, v], dim=2)     # [b, s, 3, h, d]
        # 执行内部注意力机制
        context = self.inner_attn(qkv)
        # 输出投影
        out = self.out_proj(context.view(b, s, d))
        # 应用输出投影 Dropout
        out = self.proj_drop(out)

        # 返回自注意力机制的输出
        return out


@dataclass
class ModulationOut:
    """
    调制输出类，用于存储调制参数。

    属性:
        shift (Tensor): 偏移量张量。
        scale (Tensor): 缩放因子张量。
        gate (Tensor): 门控张量。
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    调制模块，用于对输入向量进行缩放、偏移和门控。

    该模块通过一个线性层和 SiLU 激活函数生成调制参数，包括偏移量、缩放因子和门控。
    如果启用了双重调制，则生成两组调制参数。
    """
    def __init__(self, dim: int, double: bool):
        """
        初始化 Modulation 模块。

        参数:
            dim (int): 输入向量的维度。
            double (bool): 是否进行双重调制。
        """
        super().__init__()
        # 是否进行双重调制
        self.is_double = double
        # 调制参数的倍数
        self.multiplier = 6 if double else 3
        # 线性层，用于生成调制参数
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        """
        前向传播函数，执行调制操作。

        参数:
            vec (torch.Tensor): 输入向量张量。

        返回:
            Tuple[ModulationOut, Optional[ModulationOut]]: 调制输出，包括偏移量、缩放因子和门控。
        """
        # 应用线性层和 SiLU 激活函数，并分割输出为调制参数
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]), # 第一个调制输出
            ModulationOut(*out[3:]) if self.is_double else None, # 第二个调制输出（如果启用双重调制）
        )


class DoubleStreamBlock(nn.Module):
    """
    双流注意力块，用于同时处理图像和文本流。

    该模块包含图像和文本的调制、归一化和注意力机制，以及各自的 MLP 层。
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        """
        初始化 DoubleStreamBlock。

        参数:
            hidden_size (int): 隐藏层的大小。
            num_heads (int): 多头注意力的头数。
            mlp_ratio (float): MLP 层的隐藏层大小与输入大小的比例。
            qkv_bias (bool, 可选): 在查询、键和值线性层中是否使用偏置，默认为 False。
        """
        super().__init__()

        # 计算 MLP 层的隐藏层大小
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 多头注意力的头数
        self.num_heads = num_heads
        # 隐藏层的大小
        self.hidden_size = hidden_size
        # 图像调制模块，双重调制
        self.img_mod = Modulation(hidden_size, double=True)
        # 图像归一化层
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像自注意力机制
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # 图像归一化层
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像 MLP 层
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本调制模块，双重调制
        self.txt_mod = Modulation(hidden_size, double=True)
        # 文本归一化层
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本自注意力机制
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # 文本归一化层
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本 MLP 层
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播函数，执行双流注意力块的操作。

        参数:
            img (torch.Tensor): 输入图像张量。
            txt (torch.Tensor): 输入文本张量。
            vec (torch.Tensor): 输入向量张量。
            pe (torch.Tensor): 位置编码张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像和文本的输出张量。
        """
        # 图像调制
        img_mod1, img_mod2 = self.img_mod(vec)
        # 文本调制
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        # 准备图像进行注意力机制
        # 图像归一化
        img_modulated = self.img_norm1(img)
        # 应用调制
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # 计算查询、键和值
        img_qkv = self.img_attn.qkv(img_modulated)
        # 重塑张量
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 归一化查询和键
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        # 准备文本进行注意力机制
        # 文本归一化
        txt_modulated = self.txt_norm1(txt)
        # 应用调制
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # 计算查询、键和值
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        # 重塑张量
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 归一化查询和键
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        # 运行实际的自注意力机制
        q = torch.cat((txt_q, img_q), dim=2) # 连接查询
        k = torch.cat((txt_k, img_k), dim=2) # 连接键
        v = torch.cat((txt_v, img_v), dim=2) # 连接值

        # 计算注意力
        attn = attention(q, k, v, pe=pe)
        # 分割注意力输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # 计算图像块的输出
        # 应用门控和投影
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # 应用调制和 MLP
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        # 计算文本块的输出
        # 应用门控和投影
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # 应用调制和 MLP
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        # 返回图像和文本的输出
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    """
    单流注意力块（SingleStreamBlock）。

    该模块实现了 DiT（Diffusion Transformer）块，其中包含并行的线性层，并适配了调制接口。
    类似于多头自注意力机制，但仅处理单个流（例如图像流）。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        """
        初始化单流注意力块。

        参数:
            hidden_size (int): 隐藏层的大小。
            num_heads (int): 多头注意力的头数。
            mlp_ratio (float, 可选): MLP 层的隐藏层大小与输入大小的比例，默认为 4.0。
            qk_scale (float, 可选): 查询和键的缩放因子，如果为 None，则使用 head_dim 的 -0.5 次方。
        """
        super().__init__()
        # 隐藏层的大小
        self.hidden_dim = hidden_size
        # 多头注意力的头数
        self.num_heads = num_heads
        # 每个头的维度
        head_dim = hidden_size // num_heads
        # 查询和键的缩放因子
        self.scale = qk_scale or head_dim**-0.5

        # MLP 层的隐藏层大小
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        # 第一个线性层，用于计算查询、键、值和 MLP 输入
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        # 第二个线性层，用于投影和 MLP 输出
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # 查询和键的归一化层
        self.norm = QKNorm(head_dim)

        # 隐藏层的大小
        self.hidden_size = hidden_size
        # 预归一化层
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # MLP 的激活函数，使用近似 tanh 的 GELU
        self.mlp_act = nn.GELU(approximate="tanh")
        # 调制模块，单调制
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播函数，执行单流注意力块的操作。

        参数:
            x (torch.Tensor): 输入张量。
            vec (torch.Tensor): 输入向量张量，用于调制。
            pe (torch.Tensor): 位置编码张量。

        返回:
            torch.Tensor: 单流注意力块的输出。
        """
        # 应用调制
        mod, _ = self.modulation(vec)
        # 应用预归一化和调制
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # 分割为查询、键、值和 MLP 输入
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重塑张量
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 归一化查询和键
        q, k = self.norm(q, k, v)

        # compute attention
        # 计算注意力
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        # 计算 MLP 激活函数，连接并运行第二个线性层
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # 应用门控并添加残差连接
        return x + mod.gate * output


class LastLayer(nn.Module):
    """
    最终层（LastLayer）。

    该模块实现了模型的最终层，用于将输出映射到目标维度，并应用自适应层归一化（AdaLN）调制。
    AdaLN 调制根据输入向量动态调整归一化参数，从而增强模型的表达能力。
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        """
        初始化最终层。

        参数:
            hidden_size (int): 隐藏层的大小。
            patch_size (int): 图像块的大小。
            out_channels (int): 输出通道数。
        """
        super().__init__()
        # 最终层归一化层
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 线性层，用于映射到目标维度
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # SiLU 激活函数
        # 线性层，用于生成 AdaLN 的偏移量和缩放因子
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        前向传播函数，执行最终层的操作。

        参数:
            x (torch.Tensor): 输入张量。
            vec (torch.Tensor): 输入向量张量，用于 AdaLN 调制。

        返回:
            torch.Tensor: 最终层的输出。
        """
        # 应用 AdaLN 调制
        # 应用 SiLU 和线性层
        # 将输出分割为偏移量（shift）和缩放因子（scale）
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        
        # 应用归一化和调制
        # 应用缩放因子和偏移量
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]

        # 应用线性层
        # 将输入映射到目标维度
        x = self.linear(x)
        
        return x
