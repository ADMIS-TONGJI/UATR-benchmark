import os
import random
from torch.utils.data import Dataset


class UnderwaterAudioDataset(Dataset):
    """Dataset audio sous-marin - Retourne (audio_path, label)"""
    
    DEFAULT_ROOT = "/remote-home/share/dmb_nas2/Diallo/keshe/reproduction_datasets/deepship_5s_id"
    
    def __init__(
        self, 
        root_dir: str = None, 
        dataset_type: str = "train", 
        is_validation: bool = False, 
        val_split_ratio: float = 0.15,
        random_seed: int = 42
    ):
        self.root_dir = os.path.abspath(root_dir or self.DEFAULT_ROOT)
        self.dataset_type = dataset_type.lower()
        self.is_validation = is_validation
        self.val_split_ratio = val_split_ratio
        self.random_seed = random_seed
        
        if not (0 < val_split_ratio < 1):
            raise ValueError(f"val_split_ratio doit être dans (0,1), reçu {val_split_ratio}")
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root introuvable: {self.root_dir}")
        
        self.data_dir = os.path.join(self.root_dir, self.dataset_type)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dossier {self.dataset_type} introuvable: {self.data_dir}")
        
        self.classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.audio_info = self._collect_and_split()
    
    def _collect_and_split(self) -> list:
        """Collecte et split train/val"""
        class_map = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            label = self.class_to_idx[cls]
            
            for f in os.listdir(cls_dir):
                if f.lower().endswith(".wav"):
                    class_map[cls].append((os.path.join(cls_dir, f), label))
        
        empty = [cls for cls, items in class_map.items() if len(items) == 0]
        if empty:
            raise RuntimeError(f"Pas de .wav pour: {', '.join(empty)}")
        
        if self.dataset_type == "test":
            all_audio = []
            for items in class_map.values():
                all_audio.extend(items)
            return all_audio
        
        final = []
        random.seed(self.random_seed)
        
        for cls, items in class_map.items():
            shuffled = random.sample(items, len(items))
            val_size = int(len(shuffled) * self.val_split_ratio)
            
            if self.is_validation:
                final.extend(shuffled[:val_size])
            else:
                final.extend(shuffled[val_size:])
        
        if len(final) == 0:
            raise RuntimeError(f"Pas de fichiers .wav dans {self.data_dir}")
        
        return final
    
    def __getitem__(self, index: int):
        return self.audio_info[index]
    
    def __len__(self):
        return len(self.audio_info)
    
    def get_class_mapping(self) -> dict:
        return self.class_to_idx.copy()