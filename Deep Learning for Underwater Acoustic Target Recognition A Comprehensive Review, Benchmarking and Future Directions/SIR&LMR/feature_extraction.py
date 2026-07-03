import random
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

_librosa = None


def _lazy_import_librosa():
    global _librosa
    if _librosa is None:
        import librosa  # type: ignore
        _librosa = librosa
    return _librosa


def _load_audio_mono(path: str, sr: int, target_seconds: float) -> np.ndarray:
    librosa = _lazy_import_librosa()
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * target_seconds)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def _mel_spectrogram(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    librosa = _lazy_import_librosa()
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        power=2.0,
        center=True,
        pad_mode="reflect",
    )
    S_db = librosa.power_to_db(S, ref=1.0)
    return S_db.astype(np.float32)


def _standardize_per_sample(mel: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = mel.mean()
    std = mel.std()
    mel_norm = (mel - mean) / (std + eps)
    return mel_norm.astype(np.float32)


def _to_tensor_1chw(mel: np.ndarray) -> torch.Tensor:
    # [F, T] -> [1, F, T]
    return torch.from_numpy(mel[None, ...])


# ---------------------- LMR: Local Masking and Replicating ----------------------

def _apply_lmr_inplace(batch_mels: List[torch.Tensor], labels: List[int], cfg: Dict) -> None:
    """
    在 batch 内就地对若干样本施加 LMR。batch_mels[i] 形状为 [1, F, T]。
    cfg: {
      'p': float,  # 应用概率
      'num_patches': int,
      'max_h_ratio': float,  # 相对频率维高度上限 (0,1]
      'max_w_ratio': float,  # 相对时间维宽度上限 (0,1]
      'inter_prob': float    # 跨样本复制概率
    }
    """
    if not batch_mels:
        return
    p = float(cfg.get("p", 0.7))
    num_patches = int(cfg.get("num_patches", 2))
    max_h_ratio = float(cfg.get("max_h_ratio", 0.2))
    max_w_ratio = float(cfg.get("max_w_ratio", 0.2))
    inter_prob = float(cfg.get("inter_prob", 0.3))

    B = len(batch_mels)
    for i in range(B):
        if random.random() > p:
            continue
        x = batch_mels[i]  # [1, F, T]
        _, F, T = x.shape
        for _ in range(num_patches):
            h = max(1, int(F * random.random() * max_h_ratio))
            w = max(1, int(T * random.random() * max_w_ratio))
            top = 0 if F == h else random.randint(0, F - h)
            left = 0 if T == w else random.randint(0, T - w)
            # 选择复制源：同一张或跨样本
            if random.random() < inter_prob and B > 1:
                # 随机挑一个不同样本；为了鲁棒，允许跨类
                src_idx = random.randrange(B - 1)
                if src_idx >= i:
                    src_idx += 1
                src = batch_mels[src_idx]
            else:
                src = x
            # 源区域：靠近目标区域的邻域，避免越界
            src_top = 0 if F == h else max(0, min(F - h, top + random.randint(-h, h)))
            src_left = 0 if T == w else max(0, min(T - w, left + random.randint(-w, w)))
            # 使用 clone() 防止源与目标区域内存重叠导致的原地写入错误
            patch = src[:, src_top : src_top + h, src_left : src_left + w].clone()
            # 目标区域先 mask 后 replicate
            x[:, top : top + h, left : left + w] = 0.0
            x[:, top : top + h, left : left + w] = patch


# ---------------------- SIR: simulated counterpart generation ----------------------

def _generate_simulated_mel(mel: torch.Tensor, cfg: Dict) -> torch.Tensor:
    x = mel.clone()
    _, F, T = x.shape
    # 幅度缩放
    amp_scale = float(cfg.get("amp_scale", 0.1))  # +/-10%
    scale = 1.0 + random.uniform(-amp_scale, amp_scale)
    x = x * scale

    # 时间平移
    max_shift_t = int(T * float(cfg.get("max_time_shift_ratio", 0.02)))  # 2%
    if max_shift_t > 0:
        dt = random.randint(-max_shift_t, max_shift_t)
        if dt != 0:
            if dt > 0:
                x = torch.cat([x[:, :, dt:], x[:, :, :dt]], dim=2)
            else:
                dt = -dt
                x = torch.cat([x[:, :, -dt:], x[:, :, :-dt]], dim=2)

    # 频率维微小移位
    max_shift_f = int(F * float(cfg.get("max_freq_shift_ratio", 0.02)))
    if max_shift_f > 0:
        df = random.randint(-max_shift_f, max_shift_f)
        if df != 0:
            x = torch.roll(x, shifts=df, dims=1)

    # 轻微高斯平滑
    if bool(cfg.get("gauss_blur", True)):
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=x.dtype, device=x.device)
        # 频率方向
        pad_f = torch.nn.ReplicationPad2d((0, 0, 1, 1))
        xf = pad_f(x)
        xf = 0.25 * xf[:, :-2, :] + 0.5 * xf[:, 1:-1, :] + 0.25 * xf[:, 2:, :]
        # 时间方向
        pad_t = torch.nn.ReplicationPad2d((1, 1, 0, 0))
        xt = pad_t(xf)
        x = 0.25 * xt[:, :, :-2] + 0.5 * xt[:, :, 1:-1] + 0.25 * xt[:, :, 2:]

    # 轻微噪声
    noise_std = float(cfg.get("noise_std", 0.01))
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    return x


# ---------------------- Collate callable (picklable) ----------------------

class CollateMelSIRLMR:
    """可序列化的 collate，可用于 Windows 多进程 DataLoader。

    根据配置生成 Mel 特征；训练时应用 LMR，并生成 SIR 特征。
    """

    def __init__(
        self,
        *,
        is_train: bool,
        sr: int,
        target_seconds: float,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        lmr_cfg: Dict,
        sir_sim_cfg: Dict,
    ) -> None:
        self.is_train = bool(is_train)
        self.sr = int(sr)
        self.target_seconds = float(target_seconds)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.lmr_cfg = dict(lmr_cfg or {})
        self.sir_sim_cfg = dict(sir_sim_cfg or {})

    def __call__(self, batch: List[Tuple[str, int]]):
        mels: List[torch.Tensor] = []
        labels: List[int] = []
        for path, label in batch:
            y = _load_audio_mono(path, sr=self.sr, target_seconds=self.target_seconds)
            mel = _mel_spectrogram(
                y,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )
            mel = _standardize_per_sample(mel)
            mels.append(_to_tensor_1chw(mel))
            labels.append(int(label))

        if self.is_train and self.lmr_cfg.get("p", 0.0) > 0.0:
            _apply_lmr_inplace(mels, labels, self.lmr_cfg)

        feats = torch.stack(mels, dim=0)
        labels_t = torch.tensor(labels, dtype=torch.long)

        if not self.is_train:
            return feats, labels_t

        sim_list = [_generate_simulated_mel(x, self.sir_sim_cfg) for x in mels]
        feats_sim = torch.stack(sim_list, dim=0)
        return feats, labels_t, feats_sim


# ---------------------- Collate function factory ----------------------

def get_collate_fn(
    feature: str = "mel",
    is_train: bool = True,
    sr: int = 16000,
    target_seconds: float = 5.0,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    lmr_cfg: Dict = None,
    sir_sim_cfg: Dict = None,
) -> Callable:
    """
    返回 DataLoader 的 collate 可调用对象。
    - 训练: 返回 (features, labels, features_sim)
    - 验证/测试: 返回 (features, labels)
    """
    if feature.lower() != "mel":
        raise NotImplementedError(f"当前仅支持 Mel，收到: {feature}")

    return CollateMelSIRLMR(
        is_train=is_train,
        sr=sr,
        target_seconds=target_seconds,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        lmr_cfg=lmr_cfg or {},
        sir_sim_cfg=sir_sim_cfg or {},
    )
