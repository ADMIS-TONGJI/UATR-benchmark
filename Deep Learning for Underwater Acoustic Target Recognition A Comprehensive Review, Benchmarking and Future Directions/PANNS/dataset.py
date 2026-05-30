import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class UnderwaterAudioDataset(Dataset):
    """
    水声多分类任务数据集类

    文件夹结构约定：
    root_dir
    ├─ train
    │  ├─ class_a
    │  ├─ class_b
    │  └─ ...
    └─ test
       ├─ class_a
       ├─ class_b
       └─ ...
    """

    def __init__(
        self,
        root_dir: str,
        dataset_type: str = "train",
        is_validation: bool = False,
        val_split_ratio: float = 0.15,
        random_seed: int = 42,
        target_sample_rate: int = 16000,
        target_duration: float = 5.0
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.dataset_type = dataset_type.lower()
        self.is_validation = is_validation
        self.val_split_ratio = val_split_ratio
        self.random_seed = random_seed

        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_sample_rate * target_duration)

        if self.dataset_type not in ["train", "test"]:
            raise ValueError(f"dataset_type 只能是 'train' 或 'test'，当前为: {dataset_type}")

        if not (0 < val_split_ratio < 1):
            raise ValueError(f"验证集比例必须在(0, 1)之间，当前值: {val_split_ratio}")

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"根文件夹不存在：{self.root_dir}")

        self.data_dir = os.path.join(self.root_dir, self.dataset_type)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"{self.dataset_type}文件夹不存在：{self.data_dir}")

        self.classes = [
            dir_name for dir_name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, dir_name))
        ]
        self.classes.sort()

        if len(self.classes) == 0:
            raise RuntimeError(f"在 {self.data_dir} 下未找到任何类别子文件夹")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.audio_info = self._collect_and_split_audio_paths()

    def _collect_and_split_audio_paths(self) -> list:
        class_audio_map = {cls_name: [] for cls_name in self.classes}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            cls_label = self.class_to_idx[cls_name]

            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(".wav"):
                    audio_path = os.path.join(cls_dir, file_name)
                    class_audio_map[cls_name].append((audio_path, cls_label))

        empty_classes = [cls for cls, items in class_audio_map.items() if len(items) == 0]
        if empty_classes:
            raise RuntimeError(f"以下类别文件夹中未找到.wav文件：{', '.join(empty_classes)}")

        if self.dataset_type == "test":
            all_audio = []
            for items in class_audio_map.values():
                all_audio.extend(items)
            return all_audio


        final_audio_info = []
        random.seed(self.random_seed)

        for cls_name, items in class_audio_map.items():
            shuffled_items = random.sample(items, len(items))
            val_size = int(len(shuffled_items) * self.val_split_ratio)

            if len(shuffled_items) > 1 and val_size == 0:
                val_size = 1

            if self.is_validation:
                final_audio_info.extend(shuffled_items[:val_size])
            else:
                final_audio_info.extend(shuffled_items[val_size:])

        if len(final_audio_info) == 0:
            raise RuntimeError(f"在 {self.data_dir} 下未找到任何有效的.wav文件")

        return final_audio_info

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)  # (channels, samples)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # -> (samples,)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sample_rate
            )

        if waveform.size(0) < self.target_length:
            pad_len = self.target_length - waveform.size(0)
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:self.target_length]

        return waveform

    def __getitem__(self, index: int):
        audio_path, class_label = self.audio_info[index]
        waveform = self._load_audio(audio_path)
        return waveform, class_label

    def __len__(self) -> int:
        return len(self.audio_info)

    def get_class_mapping(self) -> dict:
        return self.class_to_idx.copy()

    def get_split_info(self) -> dict:
        if self.dataset_type != "train":
            return {"type": self.dataset_type, "size": len(self)}

        class_counts = {}
        for _, label in self.audio_info:
            cls_name = self.classes[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return {
            "type": "validation" if self.is_validation else "train",
            "split_ratio": self.val_split_ratio,
            "total_size": len(self),
            "class_distribution": class_counts
        }