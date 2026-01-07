import os
import numpy as np
import torch
from typing import List, Tuple
from PIL import Image

try:
    import librosa
    _HAS_LIBROSA = True
except:
    import torchaudio
    _HAS_LIBROSA = False


def _load_audio(path: str, sr: int = 32000) -> np.ndarray:
    """Charge audio"""
    if _HAS_LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    else:
        waveform, file_sr = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if file_sr != sr:
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy().astype(np.float32)


def _compute_mel_spectrogram(
    audio_path: str,
    sr: int = 32000,
    n_mels: int = 224,
    target_width: int = 224,
) -> np.ndarray:
    """
    Compute MEL spectrogram optimisé pour DINOv2
    
    Returns:
        mel_db: [n_mels, target_width] en dB normalisé
    """
    # Paramètres
    win_length = int(0.05 * sr)  # 50ms
    hop_length = int(0.025 * sr)  # 25ms
    n_fft = win_length
    
    # Charger audio
    y = _load_audio(audio_path, sr)
    
    # MEL spectrogram
    if _HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, 
            hop_length=hop_length, win_length=win_length,
            n_mels=n_mels, fmin=100, fmax=8000, 
            power=2.0, window="hann"
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
    else:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        spec = torch.stft(
            y_t, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=torch.hann_window(win_length), 
            return_complex=True
        )
        power = (spec.abs() ** 2).squeeze(0)
        
        fb = torchaudio.functional.create_fb_matrix(
            n_freqs=(n_fft // 2) + 1,
            f_min=100, f_max=8000,
            n_mels=n_mels, sample_rate=sr
        )
        mel = torch.matmul(fb, power).numpy()
        mel_db = 10 * np.log10(mel + 1e-10)
    
    # Resize à target_width
    if mel_db.shape[1] < target_width:
        # Pad
        pad = target_width - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant', constant_values=mel_db.min())
    elif mel_db.shape[1] > target_width:
        # Crop au centre
        start = (mel_db.shape[1] - target_width) // 2
        mel_db = mel_db[:, start:start + target_width]
    
    # Normalisation [0, 255] pour image
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_db = (mel_db * 255).astype(np.uint8)
    
    return mel_db


class CollateDINOv2:
    """Collate function pour DINOv2"""
    
    def __init__(
        self,
        img_size: int = 224,
        dataset_name: str = "deepship_5s_id"
    ):
        self.img_size = img_size
        self.dataset_name = dataset_name
        
        print(f"📊 CollateDINOv2: Image size {img_size}x{img_size}")
    
    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: [(audio_path, label), ...]
            
        Returns:
            images: [B, 3, H, W] RGB images
            labels: [B]
        """
        images = []
        labels = []
        
        for audio_path, label in batch:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio introuvable: {audio_path}")
            
            # Compute MEL spectrogram
            mel_db = _compute_mel_spectrogram(
                audio_path, 
                n_mels=self.img_size,
                target_width=self.img_size
            )
            
            # Convertir en image RGB (grayscale → RGB)
            mel_rgb = np.stack([mel_db, mel_db, mel_db], axis=0)  # [3, H, W]
            
            images.append(mel_rgb)
            labels.append(label)
        
        # Stack
        images = np.stack(images, axis=0)  # [B, 3, H, W]
        
        # Normalisation ImageNet (requis pour DINOv2)
        images = images.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images = (images - mean) / std
        
        images_t = torch.from_numpy(images).float()
        labels_t = torch.tensor(labels, dtype=torch.long)
        
        return images_t, labels_t