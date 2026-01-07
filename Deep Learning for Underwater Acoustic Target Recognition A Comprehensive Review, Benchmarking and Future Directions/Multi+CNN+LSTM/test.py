import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
from typing import Tuple

from dataset import UnderwaterAudioDataset
from multi_feature import collate_fn  
from model import CNN_LSTM_Model  

# -------------------------- 1. 超参数配置 --------------------------
HPARAMS = {
    # 设备配置
    "device_id": "0",
    # 数据路径
    "dataset_root": "./deepship",
    # 模型配置
    "checkpoint_path": "./checkpoints/best_val_model.pth",  # CNN-LSTM训练权重路径
    "num_classes": 4,  # 与数据集类别数
    # 推理参数
    "batch_size": 32,  
    # 结果保存
    "result_dir": "./test_results",  # 结果目录
    "cm_fig_name": "confusion_matrix.png"  # 混淆矩阵文件名
}

# -------------------------- 2. 设备选择--------------------------
def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备（推理速度较慢，建议使用GPU）")
    return device

DEVICE = get_device(HPARAMS["device_id"])

# -------------------------- 3. 加载测试数据集--------------------------
def load_test_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, dict, dict]:
    """加载测试集，返回DataLoader、类别映射（名→索引）、索引→类别名"""
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",  
        is_validation=False  
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        collate_fn=collate_fn,  
        num_workers=4,  
        pin_memory=True if DEVICE.type == "cuda" else False 
    )
    
    # 获取类别映射
    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    if len(test_dataset) > 0:
        sample_path, _ = test_dataset[0]
        sample_feature, _ = collate_fn([(sample_path, 0)])  
        feature_shape = sample_feature[0].shape  
    else:
        feature_shape = "未知（测试集无样本）"
    
    # 打印测试集关键信息（修正特征类型描述）
    print(f"\n=== 测试集信息 ===")
    print(f"特征类型: 多特征融合（梅尔+MFCC+色谱图+频谱对比度+调网络）")
    print(f"融合特征维度: {feature_shape}")
    print(f"测试集总样本数: {len(test_dataset)}")
    print(f"类别映射（名称→标签）: {class_to_idx}")
    
    return test_loader, class_to_idx, idx_to_class

# -------------------------- 4. 初始化CNN-LSTM并加载权重--------------------------
def init_model(num_classes: int, checkpoint_path: str, device: torch.device) -> CNN_LSTM_Model:
    """初始化CNN-LSTM模型"""
    model = CNN_LSTM_Model(
        num_classes=num_classes
    )
    model.to(device)
    model.eval()  # 推理模式：关闭Dropout、冻结BatchNorm统计量
    print(f"已初始化CNN-LSTM模型，设备: {device}")
    
    # 检查训练权重文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CNN-LSTM训练权重文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 提取模型状态字典
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    if next(iter(state_dict.keys()), "").startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # 加载权重到CNN-LSTM模型
    model.load_state_dict(state_dict)
    print(f"\n已加载CNN-LSTM训练权重: {os.path.basename(checkpoint_path)}")
    print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
    print(f"训练时最优验证准确率: {checkpoint.get('best_val_acc', 0.0):.4f}" if "best_val_acc" in checkpoint else "")
    
    return model

# -------------------------- 5. 模型推理 --------------------------
@torch.no_grad()  # 禁用梯度计算，减少内存占用并加速推理
def infer(model: CNN_LSTM_Model, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """推理测试集，返回所有样本的真实标签和预测标签（适配CNN-LSTM输入维度）"""
    y_true = []  # 存储真实标签
    y_pred = []  # 存储预测标签
    
    print(f"\n=== 开始推理测试集 ===")
    for batch_idx, (features, labels) in enumerate(test_loader):
        # 1. 数据移至目标设备 + 维度调整：(batch, 192, 1) → (batch, 1, 192)
        # CNN-LSTM的Conv1d要求输入格式为(batch, in_channels, seq_len)，in_channels=1
        features = features.to(device, non_blocking=True).permute(0, 2, 1)
        labels = labels.to(device, non_blocking=True)
        
        # 2. 前向传播
        _, probabilities = model(features)
        pred_labels = torch.argmax(probabilities, dim=1)  # 取概率最大的类别为预测结果
        
        # 3. 收集标签
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())
        
        # 打印推理进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  已处理批次: {batch_idx+1:3d}/{len(test_loader)}")
    
    # 转为numpy数组格式
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"推理完成！共处理样本数: {len(y_true)}")
    
    return y_true, y_pred

# -------------------------- 6. 计算评估指标 --------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict) -> Tuple[dict, str, np.ndarray]:
    """计算整体评估指标、各类别详细指标、混淆矩阵"""
    # 按索引排序类别名称
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    
    # 1. 整体指标
    overall_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
     
    # 2. 混淆矩阵（行：真实类别，列：预测类别）
    cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class.keys()))
    
    return overall_metrics,  cm

# -------------------------- 7. 混淆矩阵可视化 --------------------------
def plot_confusion_matrix(cm: np.ndarray, idx_to_class: dict, save_path: str, cmap: str = "Blues"):
    """绘制混淆矩阵热力图并保存"""
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)
    
    # 设置画布大小
    fig_size = (12 + num_classes//3, 10 + num_classes//3) if num_classes > 5 else (12, 10)
    plt.figure(figsize=fig_size)
    
    # 绘制热力图
    sns.heatmap(
        cm,
        annot=True,  # 标注每个单元格的数值
        fmt="d",  # 数值格式：整数
        cmap=cmap,  # 配色方案
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 8 if num_classes > 5 else 10}  # 按类别数调整标注字体大小
    )
    
    # 设置标签和标题
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title(f"CNN-LSTM 测试集混淆矩阵", fontsize=14, pad=20)
    
    # 旋转x轴标签，避免类别名称过长导致重叠
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # 调整布局，确保标签完整显示
    plt.tight_layout()
    
    # 保存图片（创建目录，避免路径不存在报错）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight") 
    plt.close()
    print(f"\n混淆矩阵已保存至: {os.path.abspath(save_path)}")

# -------------------------- 8. 打印评估结果 --------------------------
def print_results(overall_metrics: dict, cm: np.ndarray, idx_to_class: dict):
    """打印所有评估结果，格式清晰（标注CNN-LSTM）"""
    print("\n" + "="*90)
    print("                        CNN-LSTM 模型 - 测试集评估结果")
    print("="*90)
    
    # 1. 打印整体指标
    print("\n【1. 整体评估指标】")
    print(f"准确率 (Accuracy):    {overall_metrics['accuracy']:.4f}")
    print(f"加权F1分数 (F1-score):  {overall_metrics['f1-score']:.4f}")
    
    
    # 3. 打印文本格式混淆矩阵
    print("\n【2. 混淆矩阵】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    # 打印列标题（截取前8个字符，避免过长）
    print(" " * 18 + " ".join([f"{name[:8]:^8s}" for name in class_names]))
    # 打印每行（行标题+混淆矩阵数值）
    for i, idx in enumerate(sorted(idx_to_class.keys())):
        cls_name = idx_to_class[idx]
        row_data = cm[i]
        print(f"{cls_name[:18]:<18s}" + " ".join([f"{val:^8d}" for val in row_data]))

# -------------------------- 9. 主函数 --------------------------
def main():
    # 1. 加载测试数据
    test_loader, class_to_idx, idx_to_class = load_test_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )
    
    # 2. 初始化CNN-LSTM模型并加载权重
    model = init_model(
        num_classes=HPARAMS["num_classes"],
        checkpoint_path=HPARAMS["checkpoint_path"],
        device=DEVICE
    )
    
    # 3. 推理测试集，获取真实标签和预测标签
    y_true, y_pred = infer(model=model, test_loader=test_loader, device=DEVICE)
    
    # 4. 计算评估指标
    overall_metrics, cm = calculate_metrics(y_true, y_pred, idx_to_class)
    
    # 5. 打印评估结果
    print_results(overall_metrics, cm, idx_to_class)
    
    # 6. 绘制并保存混淆矩阵
    cm_save_path = os.path.join(HPARAMS["result_dir"], HPARAMS["cm_fig_name"])
    plot_confusion_matrix(cm, idx_to_class, cm_save_path)
    
    print("\n" + "="*90)
    print("CNN-LSTM 模型测试流程全部完成！")
    print(f"结果保存目录: {os.path.abspath(HPARAMS['result_dir'])}")
    print("="*90)

if __name__ == "__main__":
    main()