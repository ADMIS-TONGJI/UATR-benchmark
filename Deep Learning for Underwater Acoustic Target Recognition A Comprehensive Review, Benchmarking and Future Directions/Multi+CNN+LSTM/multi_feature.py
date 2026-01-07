import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft, resample, hann
from scipy.fft import dct, fft
import math
import torch

def hz_to_mel(hz):
    """将赫兹频率转换为梅尔频率"""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """将梅尔频率转换为赫兹频率"""
    return 700 * (10 **(mel / 2595) - 1)

def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """创建梅尔滤波组"""
    if fmax is None:
        fmax = sr / 2  # 奈奎斯特频率
    
    # 计算梅尔刻度上的中心频率
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_centers = np.linspace(mel_min, mel_max, n_mels + 2)  # 包含两端的边界
    hz_centers = mel_to_hz(mel_centers)  # 转换回赫兹
    
    # 将中心频率映射到FFT bins
    fft_bins = np.floor((n_fft + 1) * hz_centers / sr).astype(int)
    
    # 创建三角滤波器组
    filterbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        left = fft_bins[m - 1]
        center = fft_bins[m]
        right = fft_bins[m + 1]
        
        # 左斜率
        if center != left:
            filterbank[m - 1, left:center] = np.linspace(0, 1, center - left)
        # 右斜率
        if center != right:
            filterbank[m - 1, center:right] = np.linspace(1, 0, right - center)
    return filterbank

def extract_and_fuse_features(audio_path):
    """
    提取音频的五类特征并融合
    """
    try:
        # 1. 加载音频并调整采样率至16kHz
        sr_original, y = wav.read(audio_path)
        # 转换为单声道（如果是立体声）
        if len(y.shape) == 2:
            y = np.mean(y, axis=1)
        # 归一化到[-1, 1]
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
        # 调整采样率至16kHz
        target_sr = 16000
        if sr_original != target_sr:
            num_samples = int(len(y) * target_sr / sr_original)
            y = resample(y, num_samples)
        sr = target_sr
        # print(f"成功加载音频，采样率: {sr}Hz，时长: {len(y)/sr:.2f}s")
        
        n_fft = 2048  # FFT窗口大小
        hop_length = 512  # 帧移（步长）
        win_length = n_fft  # 窗口长度
        window = hann(win_length, sym=False)  # 汉宁窗
        
        # 3. 计算STFT（短时傅里叶变换）
        f, t, Zxx = stft(
            y,
            fs=sr,
            window=window,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            return_onesided=True
        )
        stft_mag = np.abs(Zxx)  # 取幅度谱
        stft_power = stft_mag** 2  # 功率谱（能量）
        
        # 4. 提取各类特征
        # 4.1 梅尔频谱图特征 (128,1)
        mel_filterbank = create_mel_filterbank(sr, n_fft, n_mels=128)
        mel_spectrogram = np.dot(mel_filterbank, stft_power)  # 应用梅尔滤波
        mel_feature = np.mean(mel_spectrogram, axis=1, keepdims=True)
        # print(f"梅尔频谱图特征形状: {mel_feature.shape}")
        
        # 4.2 MFCC特征 (40,1)
        log_mel = np.log10(mel_spectrogram + 1e-10)  # 加小值避免log(0)
        mfcc = dct(log_mel, type=2, axis=0, norm='ortho')  # 离散余弦变换
        mfcc = mfcc[:40, :]  # 取前40维
        mfcc_feature = np.mean(mfcc, axis=1, keepdims=True)
        # print(f"MFCC特征形状: {mfcc_feature.shape}")
        
        # 4.3 色谱图特征 (12,1)
        f_hz = f  # STFT的频率轴
        chroma = np.zeros((12, len(t)))  # 12个音高等级，时间帧数与STFT一致
        
        for i, freq in enumerate(f_hz):
            if freq < 20:  
                continue
            # 计算频率对应的音高等级（十二平均律）
            # 公式：12 * log2(freq / 440) + 69 → 转换为MIDI音高，再取模12
            midi_note = 12 * np.log2(freq / 440) + 69
            chroma_bin = int(round(midi_note) % 12)
            chroma[chroma_bin, :] += stft_power[i, :]  # 累加该频率的能量到对应音高
        
        # 归一化每个时间帧的色谱图能量
        chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)
        chroma_feature = np.mean(chroma, axis=1, keepdims=True)
        # print(f"色谱图特征形状: {chroma_feature.shape}")
        
        # 4.4 频谱对比度特征 (6,1)
        n_bands = 5  # 5个频段 → 6个特征（n_bands+1）
        min_freq = 20
        max_freq = sr / 2
        # 按八度划分频段边界（每个频段是上一个的2倍）
        band_edges = np.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            n_bands + 2,
            base=10
        )
        # 计算每个频段的掩码
        band_masks = []
        for i in range(n_bands + 1):
            mask = (f_hz >= band_edges[i]) & (f_hz < band_edges[i + 1])
            band_masks.append(mask)
        
        # 计算每个频段的能量峰值和谷值
        contrast = np.zeros((n_bands + 1, len(t)))
        for t_idx in range(len(t)):
            frame_energy = stft_power[:, t_idx]
            # 全局谷值（所有频段的最小值）
            global_valley = np.min(frame_energy[frame_energy > 1e-10]) if np.any(frame_energy > 1e-10) else 1e-10
            
            for b in range(n_bands + 1):
                mask = band_masks[b]
                band_energy = frame_energy[mask]
                if len(band_energy) == 0 or np.max(band_energy) < 1e-10:
                    contrast[b, t_idx] = 0.0
                else:
                    # 对比度 = 10*log10(峰值能量 / 谷值能量)
                    peak_energy = np.max(band_energy)
                    contrast[b, t_idx] = 10 * np.log10(peak_energy / global_valley)
        
        contrast_feature = np.mean(contrast, axis=1, keepdims=True)
        # print(f"频谱对比度特征形状: {contrast_feature.shape}")
        
        # 4.5 调网络特征 (6,1)
        low_freq_mask = f_hz < 500  
        low_energy = stft_power[low_freq_mask, :]
        if low_energy.size == 0:
            tonnetz_feature = np.zeros((6, 1))
        else:
            # 6个统计量：均值、方差、最大值、最小值、峰度、偏度
            tonnetz = np.array([
                np.mean(low_energy, axis=0),
                np.var(low_energy, axis=0),
                np.max(low_energy, axis=0),
                np.min(low_energy, axis=0) + 1e-10,
                np.mean((low_energy - np.mean(low_energy, axis=0))**4, axis=0),  # 峰度
                np.mean((low_energy - np.mean(low_energy, axis=0))**3, axis=0)   # 偏度
            ])
            tonnetz_feature = np.mean(tonnetz, axis=1, keepdims=True)
        # print(f"调网络特征形状: {tonnetz_feature.shape}")
        
        # 5. 特征融合
        fused_feature = np.concatenate(
            [mel_feature, mfcc_feature, chroma_feature, contrast_feature, tonnetz_feature],
            axis=0
        )
        
        # 验证融合后形状
        # print(f"融合后特征形状: {fused_feature.shape}")
        assert fused_feature.shape == (192, 1), f"融合特征维度错误，实际为{fused_feature.shape}"
        
        return fused_feature
    
    except Exception as e:
        print(f"处理音频时出错: {str(e)}")
        return None

def collate_fn(batch):
    """
    自定义数据拼接函数，批量提取融合特征并转换为张量
    输入：batch -> 数据集返回的样本列表，每个元素为 (audio_path, label) 元组
    输出：features -> 批量特征张量 (batch_size, 192, 1)
          labels -> 批量标签张量 (batch_size,)
    """
    # 1. 拆分批量中的音频路径和标签
    audio_paths, labels = zip(*batch)  # 分别获取所有路径和所有标签
    
    # 2. 批量提取融合特征，处理错误样本
    fused_features = []
    valid_labels = []  # 存储与有效特征对应的标签
    for path, label in zip(audio_paths, labels):
        # 调用特征提取函数
        feature = extract_and_fuse_features(path)
        if feature is not None:
            fused_features.append(feature)
            valid_labels.append(label)
    
    # 3. 验证有效样本数量
    if len(fused_features) == 0:
        raise ValueError("当前批量中所有音频样本特征提取失败，请检查音频文件格式或路径")
    
    # 4. 转换特征为numpy数组，再转为PyTorch张量
    # 特征形状：(batch_size, 192, 1)，符合函数返回的(192,1)单样本特征
    features_np = np.array(fused_features, dtype=np.float32)
    features = torch.from_numpy(features_np)
    
    # 5. 转换标签为PyTorch张量（分类任务用long类型）
    labels = torch.tensor(valid_labels, dtype=torch.long)
    
    return features, labels
