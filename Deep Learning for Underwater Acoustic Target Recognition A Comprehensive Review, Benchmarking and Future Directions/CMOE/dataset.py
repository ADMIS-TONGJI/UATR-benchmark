import os
import random
from torch.utils.data import Dataset


class UnderwaterAudioDataset(Dataset):
    """
    水声多分类任务的数据集类，支持训练集和验证集划分
    
    仅读取.wav文件路径和对应类别标签，不进行音频预处理
    
    文件夹结构约定：
    root_dir (shipsear9_5s)
    ├─ train
    │  ├─ Dredger          # 类别1文件夹
    │  │  ├─ xxx1.wav
    │  │  ├─ xxx2.wav
    │  │  └─ ...
    │  ├─ Fishboat         # 类别2文件夹
    │  └─ ... (其他类别)
    └─ test
        ├─ Dredger
        ├─ Fishboat
        └─ ... (其他类别)
    """
    
    def __init__(self, root_dir: str, dataset_type: str = "train", 
                 is_validation: bool = False, val_split_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        初始化数据集，支持训练集和验证集划分
        
        Args:
            root_dir: 根文件夹路径（如 shipsear9_5s 的路径）
            dataset_type: 数据集类型，可选 "train" 或 "test"
            is_validation: 是否为验证集，仅当dataset_type为"train"时有效
            val_split_ratio: 验证集占训练集的比例，超参数，默认为0.15
            random_seed: 随机种子，确保划分结果可重现
        """
        # 1. 验证输入参数有效性
        self.root_dir = os.path.abspath(root_dir)
        self.dataset_type = dataset_type.lower()
        self.is_validation = is_validation
        self.val_split_ratio = val_split_ratio
        self.random_seed = random_seed
        
        # 验证超参数有效性
        if not (0 < val_split_ratio < 1):
            raise ValueError(f"验证集比例必须在(0, 1)之间，当前值: {val_split_ratio}")
        
        # 检查根文件夹是否存在
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"根文件夹不存在：{self.root_dir}")
        
        # 构建train/test文件夹路径并检查
        self.data_dir = os.path.join(self.root_dir, self.dataset_type)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"{self.dataset_type}文件夹不存在：{self.data_dir}")
        
        # 2. 获取所有类别（即data_dir下的子文件夹名称）
        self.classes = [
            dir_name for dir_name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, dir_name))
        ]
        self.classes.sort()  # 排序保证类别顺序一致
        
        # 3. 建立类别名到数字标签的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 4. 收集并划分数据
        self.audio_info = self._collect_and_split_audio_paths()

    def _collect_and_split_audio_paths(self) -> list:
        """
        收集所有.wav文件的路径，并按类别划分训练集和验证集
        
        Returns:
            list: 每个元素为tuple (audio_path, class_label)
        """
        # 先按类别收集所有音频文件
        class_audio_map = {cls_name: [] for cls_name in self.classes}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            cls_label = self.class_to_idx[cls_name]
            
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(".wav"):
                    audio_path = os.path.join(cls_dir, file_name)
                    class_audio_map[cls_name].append((audio_path, cls_label))
        
        # 检查是否有类别没有数据
        empty_classes = [cls for cls, items in class_audio_map.items() if len(items) == 0]
        if empty_classes:
            raise RuntimeError(f"以下类别文件夹中未找到.wav文件：{', '.join(empty_classes)}")
        
        # 如果是测试集或不需要划分验证集，直接返回所有数据
        if self.dataset_type == "test" or not self.classes:
            all_audio = []
            for items in class_audio_map.values():
                all_audio.extend(items)
            return all_audio
        
        # 对于训练集，按类别划分训练集和验证集
        final_audio_info = []
        random.seed(self.random_seed)  # 设置随机种子，确保划分可重现
        
        for cls_name, items in class_audio_map.items():
            # 打乱每个类别的数据顺序
            shuffled_items = random.sample(items, len(items))
            
            # 计算验证集大小（每个类别单独计算）
            val_size = int(len(shuffled_items) * self.val_split_ratio)
            
            # 根据是否为验证集选择相应部分
            if self.is_validation:
                final_audio_info.extend(shuffled_items[:val_size])
            else:
                final_audio_info.extend(shuffled_items[val_size:])
        
        # 检查是否收集到有效文件
        if len(final_audio_info) == 0:
            raise RuntimeError(f"在 {self.data_dir} 下未找到任何有效的.wav文件")
        
        return final_audio_info

    def __getitem__(self, index: int) -> tuple:
        """按索引获取单个样本（文件路径 + 标签）"""
        audio_path, class_label = self.audio_info[index]
        return audio_path, class_label

    def __len__(self) -> int:
        """返回数据集总样本数"""
        return len(self.audio_info)

    def get_class_mapping(self) -> dict:
        """获取类别名到数字标签的映射字典"""
        return self.class_to_idx.copy()
    
    def get_split_info(self) -> dict:
        """获取数据集划分信息"""
        if self.dataset_type != "train":
            return {"type": self.dataset_type, "size": len(self)}
        
        # 统计每个类别的样本数量
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