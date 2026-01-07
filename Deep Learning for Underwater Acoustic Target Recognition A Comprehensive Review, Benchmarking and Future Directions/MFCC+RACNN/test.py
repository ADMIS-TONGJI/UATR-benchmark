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

# 导入自定义模块（确保路径正确）
from dataset import UnderwaterAudioDataset
from mfcc import collate_fn
from model import RACNN

# -------------------------- 1. 超参数配置 --------------------------
HPARAMS = {
    # 设备配置
    "device_id": "0",  
    # 数据路径
    "dataset_root": "/remote-home/dingyangyang/reproduction/shipsear_noise/-12dB",  
    # 模型配置
    "checkpoint_path": "/remote-home/dingyangyang/benchmark/racnn/checkpoints_s/final_epoch_model.pth",  # 加载的模型权重
    "input_dim": 2000,  # 与mfcc.py一致
    "num_classes":9,  # 类别数
    # 推理参数
    "batch_size": 16,  
    # 结果保存
    "result_dir": "./test_results",  # 评估结果保存目录
    "cm_fig_name": "confusion_matrix.png"  # 混淆矩阵图片名
}

# -------------------------- 2. 设备选择 --------------------------
def get_device(device_id: str = None) -> torch.device:
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    return device

DEVICE = get_device(HPARAMS["device_id"])

# -------------------------- 3. 加载测试数据集 --------------------------
def load_test_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, dict, dict]:
    """
    加载测试集并返回：DataLoader、类别映射（名→索引）、索引→类别名
    """
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
    
    print(f"\n测试集信息:")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {class_to_idx}")
    
    return test_loader, class_to_idx, idx_to_class

# -------------------------- 4. 初始化模型并加载权重 --------------------------
def init_model(input_dim: int, num_classes: int, checkpoint_path: str, device: torch.device) -> RACNN:
    """
    初始化RACNN模型并加载训练好的权重
    """
    # 初始化模型
    model = RACNN(input_dim=input_dim, num_classes=num_classes)
    model.to(device)
    model.eval()  # 切换到推理模式
    
    # 加载Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # 若直接保存的模型权重
        state_dict = checkpoint
    
    if next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # 加载权重
    model.load_state_dict(state_dict)
    print(f"\n模型权重已加载: {checkpoint_path}")
    print(f"加载的模型轮次: {checkpoint.get('epoch', '未知')}")
    print(f"最优验证集准确率: {checkpoint.get('best_val_acc', '未知'):.4f}" if "best_val_acc" in checkpoint else "")
    
    return model

# -------------------------- 5. 模型推理并收集标签 --------------------------
@torch.no_grad()
def infer(model: RACNN, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    推理并收集所有样本的真实标签和预测标签
    Returns:
        y_true: 真实标签数组 (num_samples,)
        y_pred: 预测标签数组 (num_samples,)
    """
    y_true = []
    y_pred = []
    
    print(f"\n开始推理测试集...")
    for batch_idx, (features, labels) in enumerate(test_loader):
        # 数据移至设备
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 推理
        _, probabilities = model(features)
        pred_labels = torch.argmax(probabilities, dim=1)
        
        # 收集标签（转为numpy数组）
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  处理批次 {batch_idx+1:3d}/{len(test_loader)}")
    
    # 转为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"推理完成！总样本数: {len(y_true)}")
    
    return y_true, y_pred

# -------------------------- 6. 计算评估指标 --------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict) -> Tuple[dict, str, np.ndarray]:
    """
    计算整体指标、各类别指标、混淆矩阵
    Returns:
        overall_metrics: 整体指标（accuracy, precision, recall, f1）
        cm: 混淆矩阵 (num_classes x num_classes)
    """
    # 类别名称列表
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)
    
    # 1. 整体指标
    overall_acc = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    overall_metrics = {
        "accuracy": overall_acc,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1-score": overall_f1
    }
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class.keys()))
    
    return overall_metrics, cm

# -------------------------- 7. 混淆矩阵可视化与保存 --------------------------
def plot_confusion_matrix(cm: np.ndarray, idx_to_class: dict, save_path: str, cmap: str = "Blues"):
    """
    绘制混淆矩阵热力图并保存
    """
    # 类别名称（按索引排序）
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    num_classes = len(class_names)
    
    # 创建画布
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,  # 标注数值
        fmt="d",  # 整数格式
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 8}  # 标注字体大小
    )
    
    # 设置标签和标题
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("混淆矩阵", fontsize=14, pad=20)
    
    # 旋转x轴标签（避免重叠）
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n混淆矩阵已保存至: {save_path}")

# -------------------------- 8. 打印评估结果 --------------------------
def print_results(overall_metrics: dict, cm: np.ndarray, idx_to_class: dict):
    """
    打印整体指标、各类别指标、混淆矩阵
    """
    print("\n" + "="*80)
    print("                          测试集评估结果")
    print("="*80)
    
    # 1. 打印整体指标
    print("\n【整体评估指标】")
    print(f"Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"F1-score (weighted):  {overall_metrics['f1-score']:.4f}")
    
    # 2. 打印混淆矩阵
    print("\n【混淆矩阵】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    # 打印列标题
    print(" " * 15 + " ".join([f"{name[:8]:8s}" for name in class_names]))
    for i, (idx, name) in enumerate(sorted(idx_to_class.items())):
        row = cm[i]
        print(f"{name[:15]:15s}" + " ".join([f"{val:8d}" for val in row]))

# -------------------------- 9. 主函数 --------------------------
def main():
    # 1. 加载测试数据
    test_loader, class_to_idx, idx_to_class = load_test_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )
    
    # 2. 初始化模型并加载权重
    model = init_model(
        input_dim=HPARAMS["input_dim"],
        num_classes=HPARAMS["num_classes"],
        checkpoint_path=HPARAMS["checkpoint_path"],
        device=DEVICE
    )
    
    # 3. 推理并收集标签
    y_true, y_pred = infer(model=model, test_loader=test_loader, device=DEVICE)
    
    # 4. 计算评估指标
    overall_metrics, cm = calculate_metrics(y_true, y_pred, idx_to_class)
    
    # 5. 打印结果
    print_results(overall_metrics, cm, idx_to_class)
    
    # 6. 绘制并保存混淆矩阵
    cm_save_path = os.path.join(HPARAMS["result_dir"], HPARAMS["cm_fig_name"])
    plot_confusion_matrix(cm, idx_to_class, cm_save_path)
    
    print("\n" + "="*80)
    print("测试完成！所有结果已输出，混淆矩阵已保存。")

if __name__ == "__main__":
    main()