import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import librosa  
from torch.utils.data import DataLoader
from dataset import UnderwaterAudioDataset


def preprocess_acoustic_signal(audio_path, frame_size=4096, normalize=True):
    """
    论文数据预处理步骤实现：
    1. 从音频文件加载时域连续声学信号
    2. 将时域连续声学信号分割为多帧（每帧 4096 个样本，无重叠）
    3. 可选数据归一化
    
    参数：
        audio_path: str，音频文件路径（支持librosa兼容格式，如wav、flac等）
        frame_size: int，帧大小，论文固定为4096
        normalize: bool，是否启用全局数据归一化（标准化：减均值除标准差）
    
    返回：
        framed_signal: torch.Tensor，形状为 (num_frames, frame_size)
                       （分帧后的数据，去掉通道维度）
    """
    # 1. 从音频文件加载时域信号（转为单声道处理）
    continuous_signal, s = librosa.load(audio_path, sr=16000)  # 返回格式：(信号数组, 采样率)，忽略采样率
    # print(s)
    # 2. 计算可分帧数量（丢弃最后不足一帧的部分）
    signal_length = len(continuous_signal)
    num_frames = signal_length // frame_size  # 整数除法取整
    valid_signal = continuous_signal[:num_frames * frame_size]  # 截取有效长度
    
    # 3. 分帧处理
    framed_signal = valid_signal.reshape(num_frames, frame_size)  # 形状：(num_frames, frame_size)
    # framed_signal = torch.tensor(framed_signal, dtype=torch.float32)  # 不再添加通道维度
    framed_signal = torch.tensor(framed_signal, dtype=torch.float32).unsqueeze(1)  # 增加通道维度，形状：(num_frames, 1, frame_size)
    
    # 4. 可选全局归一化
    if normalize:
        mean = framed_signal.mean()
        std = framed_signal.std()
        framed_signal = (framed_signal - mean) / (std + 1e-8)  # 防止除零错误
    
    return framed_signal

def collate_fn(batch):
    """
    参数：
        batch: 列表，每个元素为 (audio_path, label) 元组，其中：
               - audio_path: 音频文件路径（字符串）
               - label: 该音频对应的标签（整数）
    
    返回：
        features: torch.Tensor，形状为(total_frames, 1, frame_size)，所有样本的帧拼接
        labels: torch.Tensor，形状为(total_frames,)，与帧对应的标签
    """
    feature_list = []
    label_list = []
    
    for audio_path, label in batch:  # 注意：这里第一个元素是路径，不是预处理后的特征
        # 关键修复：对音频路径进行预处理，得到特征张量
        features = preprocess_acoustic_signal(audio_path)  # 调用预处理函数
        feature_list.append(features)
        # 为每个帧分配标签（每个帧继承原音频的标签）
        label_list.extend([label] * features.shape[0])  # 现在features是张量，有shape属性
    
    # 拼接所有帧特征，转换标签为张量
    batch_features = torch.cat(feature_list, dim=0)
    batch_labels = torch.tensor(label_list, dtype=torch.long)
    
    return batch_features, batch_labels
