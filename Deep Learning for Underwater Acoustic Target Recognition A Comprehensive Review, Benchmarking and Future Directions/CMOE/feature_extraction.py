import os
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

_HAS_LIBROSA = False
_HAS_TORCHAUDIO = False
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except Exception:
    try:
        import torchaudio  # type: ignore
        _HAS_TORCHAUDIO = True
    except Exception:
        _HAS_TORCHAUDIO = False

if not _HAS_LIBROSA and not _HAS_TORCHAUDIO:
    raise ImportError("需要 librosa 或 torchaudio 支持音频读取")

# ======================== 论文参数配置 ========================

def _get_dataset_params(dataset_name: str) -> Dict[str, Any]:
    """
    根据论文 Table 2 , Sec 4.2 , Sec 4.4  返回数据集特定参数
    """
    if dataset_name == "shipsear9_5s":
        return {
            "sr": 52734,
            "fmin": 100.0,
            "fmax": 26367.0,  # Nyquist [cite: 442]
            "n_mels": 300,
            "n_barks": 300,
            "cqt_bins": 340
        }
    elif dataset_name == "oceanship_5s":
        return {
            "sr": 32000,
            "fmin": 100.0,
            "fmax": 8000.0,  # [cite: 442]
            "n_mels": 256,
            "n_barks": 300,
            "cqt_bins": 290
        }
    elif dataset_name in ("deepship_5s_id", "deepship_5s_normal"):
        return {
            "sr": 32000,
            "fmin": 100.0,
            "fmax": 8000.0,
            "n_mels": 256,
            "n_barks": 300,
            "cqt_bins": 290
        }
    else:
        print(f"警告: 未知的 dataset_name '{dataset_name}'，使用通用默认参数 (sr=32k, 100-8000Hz)。")
        return {
            "sr": 32000,
            "fmin": 100.0,
            "fmax": 8000.0,
            "n_mels": 256,
            "n_barks": 300,
            "cqt_bins": 290
        }

# 根据论文 Table 2  的 (Time, Freq) 维度
TARGET_FRAMES_MAP = {
    "stft": 1200,
    "mel": 1200,
    "bark": 1200,
    "cqt": 900
}

# CQT 特定参数 (与论文 Sec 3.1 保持一致)
BINS_PER_OCTAVE = 24  # 假设值
CQT_FMIN_DEFAULT = 32.7 # 假设值


# ======================== 底层特征函数========================

def _load_audio(path: str, sr: int) -> np.ndarray:
    if _HAS_LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    waveform, file_sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(file_sr, sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32)


def _stft_mag(y: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    if _HAS_LIBROSA:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        return np.abs(S)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=torch.hann_window(win_length), return_complex=True)
    return spec.abs().squeeze(0).numpy()


def _hz_to_bark(f: np.ndarray) -> np.ndarray:
    # 论文公式 (1) [cite: 143]
    # 论文公式为: 6 * asinh(f / 600)
    return 6.0 * np.arcsinh(f / 600.0)


def _bark_to_hz(z: np.ndarray) -> np.ndarray:
    return 600.0 * np.sinh(z / 6.0)


def _apply_triangle_filterbank(power_linear: np.ndarray, centers_hz: np.ndarray, sr: int, n_fft: int, fmin: float, fmax: float) -> np.ndarray:
    f_bins = np.linspace(0, sr / 2, num=(n_fft // 2) + 1)
    
    # 构建滤波器组边界
    edges = np.zeros(len(centers_hz) + 2, dtype=np.float64)
    edges[1:-1] = (centers_hz[:-1] + centers_hz[1:]) / 2.0
    first_step = centers_hz[1] - centers_hz[0]
    last_step = centers_hz[-1] - centers_hz[-2]
    edges[0] = max(fmin, centers_hz[0] - first_step)
    edges[-1] = min(fmax, centers_hz[-1] + last_step)

    W = np.zeros((len(centers_hz), len(f_bins)), dtype=np.float32)
    for k in range(len(centers_hz)):
        left = edges[k]
        center = centers_hz[k]
        right = edges[k + 1]
        
        # 确保滤波器在 [fmin, fmax] 范围内
        left = max(fmin, left)
        right = min(fmax, right)

        left_mask = (f_bins >= left) & (f_bins <= center)
        right_mask = (f_bins >= center) & (f_bins <= right)
        
        if center > left:
            W[k, left_mask] = (f_bins[left_mask] - left) / (center - left)
        if right > center:
            W[k, right_mask] = (right - f_bins[right_mask]) / (right - center)
    
    # 确保只在 [fmin, fmax] 区间应用
    freq_mask = (f_bins >= fmin) & (f_bins <= fmax)
    return W[:, freq_mask] @ power_linear[freq_mask, :]


def _mel_spec(y: np.ndarray, sr: int, n_fft: int, hop_length: int, win_length: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    if _HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, window="hann"
        )
        return np.log1p(mel)  # 论文 [cite: 142] 提到了对数刻度 (logarithmic scale)
    
    # Torchaudio 实现
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=torch.hann_window(win_length), return_complex=True)
    power = (spec.abs() ** 2).squeeze(0) # [N_FFT/2 + 1, T]
    
    fb = torchaudio.functional.create_fb_matrix(
        n_freqs=(n_fft // 2) + 1, 
        f_min=fmin, 
        f_max=fmax,
        n_mels=n_mels, 
        sample_rate=sr
    )
    mel = torch.matmul(fb, power).numpy()
    return np.log1p(mel)


def _bark_spec(y: np.ndarray, sr: int, n_fft: int, hop_length: int, win_length: int, n_barks: int, fmin: float, fmax: float) -> np.ndarray:
    # 1. 计算 STFT 功率谱
    if _HAS_LIBROSA:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        power = np.abs(S) ** 2
    else:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                          window=torch.hann_window(win_length), return_complex=True)
        power = (spec.abs() ** 2).squeeze(0).numpy()
    
    # 2. 计算 Bark 滤波器组中心 (在 fmin 和 fmax 之间)
    z_min = _hz_to_bark(np.array([fmin]))[0]
    z_max = _hz_to_bark(np.array([fmax]))[0]
    centers_z = np.linspace(z_min, z_max, num=n_barks)
    centers_hz = _bark_to_hz(centers_z)
    
    # 3. 应用滤波器组
    bark = _apply_triangle_filterbank(power, centers_hz, sr, n_fft, fmin, fmax)
    return np.log1p(bark) # 论文 [cite: 142] 提到了对数刻度


def _cqt_spec(y: np.ndarray, sr: int, hop_length: int, fmin: float, n_bins: int) -> np.ndarray:
    if _HAS_LIBROSA:
        C = librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin,
                        n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE, window="hann")
        return np.log1p(np.abs(C))
    
    # Torchaudio 实现
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    cqt = torchaudio.transforms.CQT(
        sample_rate=sr, 
        hop_length=hop_length, 
        fmin=fmin,
        n_bins=n_bins, 
        bins_per_octave=BINS_PER_OCTAVE
    )(y_t)
    return np.log1p(cqt.abs().squeeze(0).numpy())


def _pad_or_trim(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """
    填充或裁剪时间维度（第二维）以匹配 target_frames
    """
    F, T = spec.shape
    if T == target_frames:
        return spec
    if T < target_frames:
        # 填充 (在时间轴右侧补零)
        return np.pad(spec, ((0, 0), (0, target_frames - T)), mode="constant")
    # 裁剪 (从中间裁剪)
    start = (T - target_frames) // 2
    end = start + target_frames
    return spec[:, start:end]


# ======================== 提取函数 ========================

def _extract_by_type(audio_path: str, feature_type: str, 
                     params: Dict[str, Any], sr: int, n_fft: int, 
                     hop_length: int, win_length: int, target_frames: int) -> np.ndarray:
    
    y = _load_audio(audio_path, sr=sr)
    
    if feature_type == "stft":
        spec = _stft_mag(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # 截断 STFT 频率到 [fmin, fmax] (近似)
        f_bins = np.linspace(0, sr / 2, num=(n_fft // 2) + 1)
        f_mask = (f_bins >= params['fmin']) & (f_bins <= params['fmax'])
        spec = spec[f_mask, :]
        
    elif feature_type == "mel":
        spec = _mel_spec(y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                         n_mels=params['n_mels'], fmin=params['fmin'], fmax=params['fmax'])
        
    elif feature_type == "bark":
        spec = _bark_spec(y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                          n_barks=params['n_barks'], fmin=params['fmin'], fmax=params['fmax'])
        
    elif feature_type == "cqt":
        # 注意: 论文中 CQT 的 fmin 是 100Hz ，而不是 32.7Hz
        cqt_fmin = params.get('cqt_fmin', params['fmin']) 
        spec = _cqt_spec(y, sr=sr, hop_length=hop_length, 
                         fmin=cqt_fmin, n_bins=params['cqt_bins'])
    else:
        raise ValueError(f"未知特征类型: {feature_type}")

    # 填充/裁剪到目标帧数
    spec = _pad_or_trim(spec, target_frames)

    # 样本内标准化
    mean = spec.mean()
    std = spec.std() + 1e-6
    spec = (spec - mean) / std
    return spec.astype(np.float32)


# ======================== 可pickle的批处理器 ========================
class CollateByFeature:
    def __init__(self, feature_type: str, dataset_name: str) -> None:
        self.feature_type = feature_type.strip().lower()
        self.dataset_name = dataset_name
        
        # 获取论文指定的参数 
        self.params = _get_dataset_params(dataset_name)
        self.sr = self.params['sr']
        
        # 帧长/帧移 (50ms, 25ms) 
        self.win_length = int(0.05 * self.sr)
        self.hop_length = int(0.025 * self.sr)
        # 将 n_fft 设置为 win_length
        self.n_fft = self.win_length 
        
        # 目标帧数 (Table 2) 
        if self.feature_type not in TARGET_FRAMES_MAP:
            raise ValueError(f"特征 {self.feature_type} 未在 TARGET_FRAMES_MAP 中定义")
        self.target_frames = TARGET_FRAMES_MAP[self.feature_type]

    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats: List[np.ndarray] = []
        labels: List[int] = []
        
        for path, label in batch:
            if not os.path.exists(path):
                raise FileNotFoundError(f"音频文件不存在: {path}")
            
            spec = _extract_by_type(
                path, 
                self.feature_type, 
                self.params, 
                self.sr, 
                self.n_fft, 
                self.hop_length, 
                self.win_length, 
                self.target_frames
            )
            
            feats.append(spec)
            labels.append(int(label))

        # 对齐频率维（高度）到最小值，避免拼接维度不一致
        min_H = min(f.shape[0] for f in feats)
        feats_aligned = []
        for f in feats:
            if f.shape[0] > min_H:
                print(f"警告: 特征高度不一致 (预期 {min_H}, 得到 {f.shape[0]})。进行裁剪。")
                feats_aligned.append(f[:min_H, :])
            else:
                feats_aligned.append(f)

        x = np.stack(feats_aligned, axis=0)[:, np.newaxis, :, :]  # [B, 1, H, W]
        X = torch.from_numpy(x)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y


def get_collate_fn(feature_type: str, dataset_name: str):
    """
    返回顶层可调用对象，避免Windows下多进程DataLoader对闭包的pickle失败
    """
    return CollateByFeature(feature_type, dataset_name)