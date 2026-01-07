import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import UnderwaterAudioDataset
import os
import math

# 配置参数
SAMPLE_RATE = 16000  # 保持与论文一致的采样率
N_FFT = 2048         # FFT窗口大小
HOP_LENGTH = 512     # 帧移，控制时间分辨率
WINDOW = 'hann'      # 窗函数
SAMPLE_LENGTH = 5 # 每个样本的长度，单位s
TARGET_FREQ_BINS = math.ceil(N_FFT/2)+1  # 频率维度 (n_fft/2 + 1)
TARGET_TIME_STEPS = math.ceil(SAMPLE_RATE*SAMPLE_LENGTH/HOP_LENGTH)  # 音频对应的时间步数 (以5s为例,16000*5/512 ≈ 156.25,向上取整)


def extract_stft(audio_path):
    """提取STFT特征"""
    # 加载音频文件，确保采样率一致
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    target_length = SAMPLE_RATE * SAMPLE_LENGTH  
    length_diff = abs(len(y) - target_length)
    if length_diff > 100:  # 允许100样本以内的误差
        raise ValueError(f"音频长度不符合要求，实际长度: {len(y)/SAMPLE_RATE:.2f}秒")
    
    if len(y) != target_length:
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
    
    # 计算STFT
    stft = librosa.stft(
        y=y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        center=True
    )
    
    # 转换为幅度谱并转为分贝值
    stft_mag = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    
    # 确保特征维度统一
    if stft_db.shape[1] < TARGET_TIME_STEPS:
        stft_db = np.pad(
            stft_db, 
            ((0, 0), (0, TARGET_TIME_STEPS - stft_db.shape[1])), 
            mode="constant"
        )
    else:
        stft_db = stft_db[:, :TARGET_TIME_STEPS]
    
    stft_db = stft_db[np.newaxis, ...]
    
    return stft_db


def collate_fn(batch):
    """数据拼接函数"""
    audio_paths, labels = zip(*batch)
    
    # 提取所有样本的STFT特征
    features = []
    for path in audio_paths:
        stft_feature = extract_stft(path)
        features.append(stft_feature)
    
    features_np = np.array(features, dtype=np.float32)  # 形状: (batch_size, 1, 1025, 157)
    features = torch.from_numpy(features_np)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels

