import os 
import torch 
import argparse
import math 
from einops import rearrange, repeat
from PIL import Image
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan

from utils import load_t5, load_clap, load_ae
from train import RF 
from constants import build_model


def prepare(t5, clip, img, prompt):
    """
    准备模型输入数据。

    该函数处理输入图像和提示文本，生成模型所需的图像张量、文本张量以及相关的标识符和嵌入向量。

    参数:
        t5: T5 模型实例，用于处理文本输入。
        clip: CLIP 模型实例，用于生成文本嵌入向量。
        img (torch.Tensor): 输入图像张量，形状为 (bs, c, h, w)。
        prompt (str 或 List[str]): 输入的提示文本，可以是单个字符串或字符串列表。

    返回:
        Tuple[torch.Tensor, dict]: 处理后的图像张量和包含相关信息的字典。
    """
    # 获取图像的批量大小 (bs)、通道数 (c)、高度 (h) 和宽度 (w)
    bs, c, h, w = img.shape

    # 如果批量大小为1且提示不是字符串，则假设提示是字符串列表，长度为批量大小
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # 重塑图像张量，从 (b, c, h, w) 变为 (b, h*w, c*ph*pw)
    # 这里 ph 和 pw 是重塑时的块大小，这里设置为 2
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # 如果图像的批量大小为1但实际批量大小大于1，则重复图像以匹配批量大小
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # 生成图像标识符张量，形状为 (h//2, w//2, 3)
    # 这里假设图像被分割成 2x2 的块，因此高度和宽度都除以 2
    img_ids = torch.zeros(h // 2, w // 2, 3)

    # 对第二个通道赋值，使其值为行索引
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    # 对第三个通道赋值，使其值为列索引
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    # 重复图像标识符张量以匹配批量大小
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # 处理提示文本
    if isinstance(prompt, str):
        # 如果提示是字符串，则转换为列表
        prompt = [prompt]

    # 使用 T5 模型处理提示文本
    txt = t5(prompt)
    # 如果文本的批量大小为1但实际批量大小大于1，则重复文本以匹配批量大小
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    # 生成文本标识符张量，形状为 (bs, txt.shape[1], 3)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    # 使用 CLIP 模型生成文本嵌入向量
    vec = clip(prompt)
    # 如果向量的批量大小为1但实际批量大小大于1，则重复向量以匹配批量大小
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    # 输出图像标识符、文本张量和向量的尺寸
    print(img_ids.size(), txt.size(), vec.size())
    # 返回处理后的图像张量和包含相关信息的字典
    return img, {
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "y": vec.to(img.device),
    }


def main(args):
    """
    主函数，用于生成音乐。

    该函数加载预训练的模型和权重，读取提示文本，生成音乐并保存为 WAV 文件。

    参数:
        args: 命令行参数，包含以下属性：
            - seed (int): 随机种子。
            - version (str): 模型版本。
            - ckpt_path (str): 模型检查点路径。
            - audioldm2_model_path (str): audioldm2 模型路径。
            - prompt_file (str): 提示文件路径。
    """

    print('generate with Music!')
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定义潜在空间的大小
    latent_size = (256, 16) 

    # 构建模型并移动到指定设备
    model = build_model(args.version).to(device) 
    local_path = args.ckpt_path
    state_dict = torch.load(local_path, map_location=lambda storage, loc: storage)
    # 加载 EMA（指数移动平均）模型的权重
    model.load_state_dict(state_dict['ema'])
    model.eval()  # important! 

    # 初始化扩散模型
    diffusion = RF()

    # Setup VAE
    # 设置 VAE（变分自编码器）
    # 加载 T5 模型，用于处理文本输入
    t5 = load_t5(device, max_length=256)
    # 加载 CLAP 模型，用于生成文本嵌入向量
    clap = load_clap(device, max_length=256)

    # 加载预训练的 VAE 模型
    vae = AutoencoderKL.from_pretrained(os.path.join(args.audioldm2_model_path, 'vae')).to(device)
    # 加载预训练的声码器模型
    vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(args.audioldm2_model_path, 'vocoder')).to(device)

    # 从提示文件中读取条件文本
    with open(args.prompt_file, 'r') as f: 
        conds_txt = f.readlines()
    
    # 获取条件文本的数量
    L = len(conds_txt) 
    # 生成无条件文本列表，这里使用 "low quality, gentle" 作为无条件文本
    unconds_txt = ["low quality, gentle"] * L 
    print(L, conds_txt, unconds_txt) 

    # 生成初始噪声张量，形状为 (L, 8, latent_size[0], latent_size[1])
    init_noise = torch.randn(L, 8, latent_size[0], latent_size[1]).cuda() 

    # 设置采样步数为 50
    STEPSIZE = 50
    # 准备模型输入数据，包括图像张量和条件
    img, conds = prepare(t5, clap, init_noise, conds_txt)
    # 准备模型输入数据，包括图像张量和无条件
    _, unconds = prepare(t5, clap, init_noise, unconds_txt) 
    
    # 使用自动混合精度进行推理
    with torch.autocast(device_type='cuda'): 
        # 使用扩散模型进行采样，生成图像
        images = diffusion.sample_with_xps(model, img, conds=conds, null_cond=unconds, sample_steps = STEPSIZE, cfg = 7.0)
    # 输出生成的图像的形状
    print(images[-1].size(), )
    
    # 重塑图像张量，从 (b, h*w, c*ph*pw) 变为 (b, c, h*ph, w*pw)
    images = rearrange(
        images[-1], 
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=128,
        w=8,
        ph=2,
        pw=2,)
    
    # 打印重塑后的图像张量形状
    # print(images.size())
    # 将图像张量缩放到 VAE 的潜在空间
    latents = 1 / vae.config.scaling_factor * images
    # 使用 VAE 解码生成梅尔频谱
    mel_spectrogram = vae.decode(latents).sample 
    print(mel_spectrogram.size()) 
    
    # 对每个样本进行处理
    for i in range(L): 
        x_i = mel_spectrogram[i]
        # 如果维度为 4，则去掉第二维
        if x_i.dim() == 4:
            x_i = x_i.squeeze(1)
        # 使用声码器将梅尔频谱转换为波形
        waveform = vocoder(x_i)
        # 将波形转换为 numpy 数组，并移动到 CPU
        waveform = waveform[0].cpu().float().detach().numpy()
        print(waveform.shape)
        # import soundfile as sf
        # sf.write('reconstruct.wav', waveform, samplerate=16000) 
        from  scipy.io import wavfile 
        wavfile.write('wav/sample_' + str(i) + '.wav', 16000, waveform) 
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="small")
    parser.add_argument("--prompt_file", type=str, default='config/example.txt')
    parser.add_argument("--ckpt_path", type=str, default='musicflow_s.pt')
    parser.add_argument("--audioldm2_model_path", type=str, default='/maindata/data/shared/multimodal/public/dataset_music/audioldm2' )
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    main(args)


