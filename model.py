from dataclasses import dataclass
import torch
from torch import Tensor, nn

from layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding


@dataclass
class FluxParams:
    """
    Flux 模型参数类，用于定义模型的各个参数。

    参数:
        in_channels (int): 输入通道数。
        vec_in_dim (int): 向量输入的维度。
        context_in_dim (int): 上下文输入的维度。
        hidden_size (int): 隐藏层的大小。
        mlp_ratio (float): MLP 层的隐藏层大小与输入大小的比例。
        num_heads (int): 多头注意力的头数。
        depth (int): 双流块（DoubleStreamBlock）的层数。
        depth_single_blocks (int): 单流块（SingleStreamBlock）的层数。
        axes_dim (List[int]): 各轴的维度列表，用于位置编码。
        theta (int): 位置编码的参数 θ。
        qkv_bias (bool): 在查询（query）、键（key）和值（value）线性层中是否使用偏置。
        guidance_embed (bool): 是否使用指导嵌入。
    """
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    """
    Flux 模型类，实现了用于序列流匹配的 Transformer 模型。

    该模型结合了图像输入、时间步长输入、向量输入和文本输入，通过多头自注意力机制和前馈神经网络进行处理。
    """

    def __init__(self, params: FluxParams):
        """
        初始化 Flux 模型。

        参数:
            params (FluxParams): 模型参数对象，包含模型的各个参数设置。
        """
        super().__init__()

        self.params = params
        # 输入通道数
        self.in_channels = params.in_channels
        # 输出通道数与输入通道数相同
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        
        # 计算位置编码的维度
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        # 隐藏层大小
        self.hidden_size = params.hidden_size
        # 多头注意力的头数
        self.num_heads = params.num_heads
        # 初始化位置编码嵌入器
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        # 图像输入线性层
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # 时间步长输入的 MLP 嵌入器
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # 向量输入的 MLP 嵌入器
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)

        # 指导输入的 MLP 嵌入器（如果使用指导嵌入）
        # self.guidance_in = (
        #    MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        # )

        # 文本输入线性层
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # 初始化双流块列表，每个双流块包含图像和文本的注意力机制
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        # 初始化单流块列表，每个单流块仅处理图像
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 初始化最终层，将输出映射到目标维度
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        x: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        t: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """
        前向传播方法，实现模型的前向计算。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (N, T, in_channels)。
            img_ids (torch.Tensor): 输入图像的 ID 张量，形状为 (N, T)。
            txt (torch.Tensor): 输入文本张量，形状为 (N, T, context_in_dim)。
            txt_ids (torch.Tensor): 输入文本的 ID 张量，形状为 (N, T)。
            t (torch.Tensor): 时间步长张量，形状为 (N, T)。
            y (torch.Tensor): 向量输入张量，形状为 (N, T, vec_in_dim)。
            guidance (Optional[torch.Tensor], 可选): 指导张量，形状为 (N, T, guidance_dim)。

        返回:
            torch.Tensor: 输出张量，形状为 (N, T, patch_size ** 2 * out_channels)。
        """
        if x.ndim != 3 or txt.ndim != 3:
            # 检查输入图像和文本张量的维度是否为3
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        # 对输入图像进行线性变换
        img = self.img_in(x)
        # 对时间步长进行嵌入
        vec = self.time_in(timestep_embedding(t, 256))

        # 如果使用指导嵌入，则添加指导嵌入
        # if self.params.guidance_embed:
        #    if guidance is None:
        #        raise ValueError("Didn't get guidance strength for guidance distilled model.")
        #    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        # 添加向量输入
        vec = vec + self.vector_in(y)
        # 对文本输入进行线性变换
        txt = self.txt_in(txt)

        # 将文本 ID 和图像 ID 连接起来
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # 生成位置编码
        pe = self.pe_embedder(ids)

        # 通过多个双流块处理图像和文本
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # 将图像和文本连接起来
        img = torch.cat((txt, img), 1)
        # 通过多个单流块处理图像
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        
        # 裁剪图像部分，仅保留图像部分
        img = img[:, txt.shape[1] :, ...]

        # 通过最终层输出
        # 输出形状为 (N, T, patch_size ** 2 * out_channels)
        img = self.final_layer(img, vec)
        return img
