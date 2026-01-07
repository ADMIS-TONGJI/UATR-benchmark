import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import UnderwaterAudioDataset

# 配置参数
N_FFT = 2048
HOP_LENGTH = 2048 - 1536  # 512
N_MELS = 50
N_MFCC = 13
FLATTENED_SIZE = 2000
SAMPLE_RATE = 16000  # 音频采样率



def extract_mfcc(audio_path):
    """提取MFCC特征"""
    # 加载音频文件，确保采样率一致
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 计算梅尔频谱
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # 转换为对数能量
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=N_MFCC
    )

    mfcc_flat = mfcc.flatten()
    # print(len(mfcc_flat))
    if len(mfcc_flat) < FLATTENED_SIZE:
        # 不足时补零
        mfcc_flat = np.pad(
            mfcc_flat, 
            (0, FLATTENED_SIZE - len(mfcc_flat)), 
            mode="constant"
        )
    else:
        # 过长时截断
        mfcc_flat = mfcc_flat[:FLATTENED_SIZE]
    
    return mfcc_flat


def collate_fn(batch):
    """数据拼接函数"""
    audio_paths, labels = zip(*batch)
    
    # 提取所有样本的MFCC特征
    features = []
    for path in audio_paths:
        mfcc = extract_mfcc(path)
        features.append(mfcc)
    
    features_np = np.array(features, dtype=np.float32)  # 形状: (batch_size, 2000)
    features = torch.from_numpy(features_np)  
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels