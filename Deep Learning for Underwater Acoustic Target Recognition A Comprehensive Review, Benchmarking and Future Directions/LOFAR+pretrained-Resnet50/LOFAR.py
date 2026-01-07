import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn.functional as F  

def compute_lofar(input_audio_path, 
                  output_image_path, 
                  target_sr=12000,
                  n_fft=1024,
                  overlap_ratio=0.75,
                  freq_min=20,
                  freq_max=3000,
                  img_size=(224, 224)):  
    # 1. 读取音频并重采样
    y, sr = librosa.load(input_audio_path, sr=target_sr)
    
    # 2. 分帧并计算STFT
    hop_length = int(n_fft * (1 - overlap_ratio))
    stft = librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann'
    )
    
    # 3. 取幅度谱
    magnitude = np.abs(stft)
    
    # 4. 频率轴处理
    frequencies = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)
    freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    magnitude = magnitude[freq_mask, :]
    # frequencies = frequencies[freq_mask]  
    
    # 5. 每帧归一化
    num_frames = magnitude.shape[1]
    for i in range(num_frames):
        frame = magnitude[:, i]
        frame_mean = np.mean(frame)
        frame_max = np.max(frame)
        frame_min = np.min(frame)
        if frame_max - frame_min < 1e-8:
            magnitude[:, i] = 0.0
        else:
            magnitude[:, i] = (frame - frame_mean) / (frame_max - frame_min)
    
    # 6. 调整动态范围（压缩并归一化到[0, 1]）
    magnitude = np.log1p(magnitude)  # 对数压缩
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-8)
    
    # 7. 使用PyTorch双线性插值调整为目标尺寸 (224, 224)
    magnitude_tensor = torch.from_numpy(magnitude).unsqueeze(0).unsqueeze(0).float()
    
    # 双线性插值调整大小
    resized_tensor = F.interpolate(
        magnitude_tensor,
        size=img_size,
        mode='bilinear', 
        align_corners=False
    )
    
    magnitude_resized = resized_tensor.squeeze(0).squeeze(0).numpy()
    

    return magnitude_resized.astype(np.float32)


def collate_fn(batch):
    """
    自定义数据拼接函数，将批次中的音频路径转换为LOFAR特征，并整理标签为张量
    
    参数:
        batch: 列表，每个元素为 (audio_path, label) 元组
        
    返回:
        features: 张量，形状为 (batch_size, 224, 224)，批次LOFAR特征
        labels: 张量，形状为 (batch_size,)，批次标签
    """
    # 从批次中分离音频路径和标签（每个样本是 (audio_path, label) 元组）
    audio_paths, labels = zip(*batch)
    
    # 提取每个音频的LOFAR特征
    features = []
    for audio_path in audio_paths:
        # 调用compute_lofar提取特征，不保存图像
        lofar_feature = compute_lofar(
            input_audio_path=audio_path,
            output_image_path=None,  # 不保存图像，仅返回特征
            target_sr=12000,
            n_fft=1024,
            overlap_ratio=0.75,
            freq_min=20,
            freq_max=3000,
            img_size=(224, 224)
        )
        features.append(lofar_feature)
    
    # 将特征列表转换为numpy数组，再转为PyTorch张量（形状：[batch_size, 224, 224]）
    features_np = np.array(features, dtype=np.float32)
    features = torch.from_numpy(features_np)
    
    # 将标签转换为长整型张量（形状：[batch_size,]）
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels
