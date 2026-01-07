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
from signal_normalize import collate_fn, preprocess_acoustic_signal 
from model import UATC_DenseNet

# -------------------------- 1. 超参数配置--------------------------
HPARAMS = {
    # 设备配置
    "device_id": "1",  # 与训练时一致的GPU编号，None则用CPU
    # 数据路径
    "dataset_root": "./shipsear",  # 与train.py保持一致
    # 模型配置
    "checkpoint_path": "./checkpoints/final_epoch_model.pth",  # UATC_DenseNet训练权重路径
    "num_classes": 9,  # 与数据集实际类别数一致
    # 推理参数
    "batch_size": 16,  # 可根据GPU内存调整
    # 结果保存
    "result_dir": "./test_results1",  # 结果保存目录
    "cm_fig_name": "confusion_matrix_noise.png"  # 混淆矩阵文件名标注模型
}

# -------------------------- 2. 设备选择 --------------------------
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

# -------------------------- 3. 加载测试数据集 --------------------------
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
        num_workers=4,  # 多线程加速数据加载
        pin_memory=True if DEVICE.type == "cuda" else False  # GPU时启用内存锁存
    )
    
    # 获取类别映射
    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    # 自动计算时域特征实际维度
    if len(test_dataset) > 0:
        sample_path, _ = test_dataset[0]
        sample_feature = preprocess_acoustic_signal(sample_path)  
        feature_shape = sample_feature.shape  # 格式：(num_frames, 1, frame_size)
    else:
        feature_shape = "未知（测试集无样本）"
    
    # 打印测试集关键信息
    print(f"\n=== 测试集信息 ===")
    print(f"特征类型: 时域分帧信号")
    print(f"特征维度: {feature_shape}")
    print(f"测试集总样本数: {len(test_dataset)}")
    print(f"类别映射（名称→标签）: {class_to_idx}")
    
    return test_loader, class_to_idx, idx_to_class

# -------------------------- 4. 初始化UATC_DenseNet并加载权重--------------------------
def init_model(num_classes: int, checkpoint_path: str, device: torch.device) -> UATC_DenseNet:
    """初始化UATC_DenseNet模型，加载训练权重并切换至推理模式"""
    # 初始化UATC_DenseNet
    model = UATC_DenseNet(
        num_classes=num_classes
    )
    model.to(device)
    model.eval()  # 推理模式：关闭Dropout、冻结BatchNorm统计量
    print(f"已初始化UATC_DenseNet模型，设备: {device}")
    
    # 检查权重文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"训练权重文件不存在: {checkpoint_path}")
    
    # 加载训练权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 提取模型状态字典
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # 兼容直接保存的模型权重
    
    if next(iter(state_dict.keys()), "").startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # 加载权重到模型
    model.load_state_dict(state_dict)
    print(f"\n已加载训练权重: {os.path.basename(checkpoint_path)}")
    print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
    print(f"训练时最优验证准确率: {checkpoint.get('best_val_acc', 0.0):.4f}" if "best_val_acc" in checkpoint else "")
    
    return model

# -------------------------- 5. 模型推理--------------------------
@torch.no_grad()  # 禁用梯度计算，减少内存占用并加速推理
def infer(model: UATC_DenseNet, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """推理测试集，返回所有样本的真实标签和预测标签"""
    y_true = []  # 存储真实标签
    y_pred = []  # 存储预测标签
    
    print(f"\n=== 开始推理测试集 ===")
    for batch_idx, (features, labels) in enumerate(test_loader):
        # 数据移至目标设备
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        _, probabilities = model(features)
        pred_labels = torch.argmax(probabilities, dim=1)  # 取概率最大的类别为预测结果
        
        # 收集标签
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())
        
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
    
    
    # 3. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class.keys()))
    
    return overall_metrics, cm

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
    plt.title(f"UATC_DenseNet 测试集混淆矩阵", fontsize=14, pad=20)
    
    # 旋转x轴标签，避免类别名称过长导致重叠
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # 调整布局，确保标签完整显示
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  
    print(f"\n混淆矩阵已保存至: {os.path.abspath(save_path)}")

# -------------------------- 8. 打印评估结果 --------------------------
def print_results(overall_metrics: dict, cm: np.ndarray, idx_to_class: dict):
    """打印所有评估结果"""
    print("\n" + "="*90)
    print("                        UATC_DenseNet 模型 - 测试集评估结果")
    print("="*90)
    
    print("\n【1. 整体评估指标】")
    print(f"准确率 (Accuracy):    {overall_metrics['accuracy']:.4f}")
    print(f"加权F1分数 (F1-score):  {overall_metrics['f1-score']:.4f}")
    
    
    print("\n【2. 混淆矩阵】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    # 打印列标题（截取前8个字符，避免过长）
    print(" " * 18 + " ".join([f"{name[:8]:^8s}" for name in class_names]))
    # 打印每行（行标题+混淆矩阵数值）
    for i, idx in enumerate(sorted(idx_to_class.keys())):
        cls_name = idx_to_class[idx]
        row_data = cm[i]
        print(f"{cls_name[:18]:<18s}" + " ".join([f"{val:^8d}" for val in row_data]))

# -------------------------- 9. 主函数--------------------------
def main():
    # 1. 加载测试数据
    test_loader, class_to_idx, idx_to_class = load_test_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )
    
    # 2. 初始化模型并加载权重
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
    print("UATC_DenseNet 模型测试流程全部完成！")
    print(f"结果保存目录: {os.path.abspath(HPARAMS['result_dir'])}")
    print("="*90)

if __name__ == "__main__":
    main()