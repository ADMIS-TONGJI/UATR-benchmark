import librosa
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from dataset import UnderwaterAudioDataset
import torch.nn.functional as F  

# 配置参数
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SAMPLE_RATE = 16000
TARGET_SHAPE = (224, 224)  


def extract_mel(audio_path):
    """提取梅尔频谱图特征"""
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 计算梅尔频谱并转换为对数能量
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)

    log_mel_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()
    
    # 使用双线性插值调整大小到224x224
    resized_tensor = F.interpolate(
        log_mel_tensor,
        size=TARGET_SHAPE,
        mode='bilinear',  
        align_corners=False
    )
    
    resized_log_mel = resized_tensor.squeeze(0).squeeze(0).numpy()
    
    return resized_log_mel.astype(np.float32)

def collate_fn(batch):
    """数据拼接函数"""
    audio_paths, labels = zip(*batch)
    
    # 提取所有样本的梅尔频谱特征
    features = []
    for path in audio_paths:
        mel = extract_mel(path)
        features.append(mel)
    features_np = np.array(features, dtype=np.float32)
    features = torch.from_numpy(features_np)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels