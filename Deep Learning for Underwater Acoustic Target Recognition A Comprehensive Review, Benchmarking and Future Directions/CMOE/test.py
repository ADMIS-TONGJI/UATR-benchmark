import os
import argparse
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
from matplotlib import font_manager

# 全局设置中文字体，避免 Matplotlib 无法显示中文导致的缺字警告
_cjk_fonts = [
    "Microsoft YaHei",  # Windows 常见
    "SimHei",           # 黑体
    "Noto Sans CJK SC", # Google Noto
    "Source Han Sans CN",
    "Arial Unicode MS"
]
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _font_name in _cjk_fonts:
    if _font_name in _available_fonts:
        plt.rcParams["font.sans-serif"] = [_font_name] + plt.rcParams.get("font.sans-serif", [])
        break
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import CMoE

# -------------------------- 1. 超参数配置 (修改) --------------------------
HPARAMS = {
    # 设备配置
    "device_id": "3",
    # 数据路径
    "dataset_root": "C:/Users/win11/Desktop/CMOE/data/shipsear9_5s",
    # 模型配置
    "checkpoint_path": "./checkpoints/best_val_model.pth",
    "num_classes": 9,
    # 特征选择：stft | mel | bark | cqt
    "feature": "mel",
    # CMoE配置
    "num_experts": 4,
    "gate_temperature": 1.0,
    "residual_scale": 1,
    # 推理参数
    "batch_size": 32,
    # 结果保存
    "result_dir": "./test_results",
    "cm_fig_name": "confusion_matrix.png"
}

def parse_args():
    parser = argparse.ArgumentParser(description="CMoE 测试入口")
    parser.add_argument("--dataset_name", type=str, default="shipsear9_5s",
                        choices=["shipsear9_5s", "oceanship_5s", "deepship_5s_id", "deepship_5s_normal"], help="选择数据集名称")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="数据集根路径")
    parser.add_argument("--device_id", type=str, default=None, help="CUDA 设备编号")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="模型权重路径")
    return parser.parse_args()

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

# -------------------------- 3. 加载测试数据集 --------------------------
def load_test_data(dataset_root: str, batch_size: int, feature: str, dataset_name: str) -> Tuple[DataLoader, dict, dict]:
    """
    加载测试集并返回：DataLoader、类别映射（名→索引）、索引→类别名
    """
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=False
    )
    
    # 传递 feature 和 dataset_name
    collate_fn = get_collate_fn(feature, dataset_name)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    print(f"\n测试集信息:")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别映射: {class_to_idx}")
    print(f"使用特征: {feature} (数据集: {dataset_name})")
    
    return test_loader, class_to_idx, idx_to_class

# -------------------------- 4. 初始化模型并加载权重 (修改) --------------------------
def init_model(num_classes: int, checkpoint_path: str, device: torch.device):
    """
    初始化CMoE模型并加载训练好的权重；根据checkpoint自动匹配是否启用残差专家。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 基于权重自动决定 residual 开关 
    has_residual_in_ckpt = any(k.startswith("residual_expert.") for k in state_dict.keys())

    # 更新 CMoE 实例化
    model = CMoE(
        num_classes=num_classes,
        num_experts=HPARAMS["num_experts"],
        gate_temperature=HPARAMS["gate_temperature"],
        residual=has_residual_in_ckpt,
        residual_scale=HPARAMS["residual_scale"]
    )
    
    print(f"使用CMoE模型进行推理 (Residual={has_residual_in_ckpt})：K={HPARAMS['num_experts']}")
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict)
    print(f"\n模型权重已加载: {checkpoint_path}")
    print(f"加载的模型轮次: {checkpoint.get('epoch', '未知')}")
    print(f"最优验证集准确率: {checkpoint.get('best_val_acc', '未知'):.4f}" if "best_val_acc" in checkpoint else "")

    return model

# -------------------------- 5. 模型推理并收集标签 --------------------------
@torch.no_grad()
def infer(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    推理并收集所有样本的真实标签和预测标签
    """
    y_true = []
    y_pred = []
    
    print(f"\n开始推理测试集...")
    for batch_idx, (features, labels) in enumerate(test_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 推理 (论文 Sec 3.3 [cite: 220, 285])
        # CMoE 返回 (logits, probabilities, aux_losses)
        _, probabilities, _ = model(features)
        pred_labels = torch.argmax(probabilities, dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred_labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  处理批次 {batch_idx+1:3d}/{len(test_loader)}")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"推理完成！总样本数: {len(y_true)}")
    
    return y_true, y_pred

# -------------------------- 6. 计算评估指标 (论文 Sec 5 [cite: 359]) --------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: dict) -> Tuple[dict, str, np.ndarray]:
    """
    计算整体指标、各类别指标、混淆矩阵
    """
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    
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
    
    cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class.keys()))
    
    return overall_metrics,  cm

# -------------------------- 7. 混淆矩阵可视化与保存 (论文 Fig 5 [cite: 506]) --------------------------
def plot_confusion_matrix(cm: np.ndarray, idx_to_class: dict, save_path: str, cmap: str = "Blues"):
    """
    绘制混淆矩阵热力图并保存
    """
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 8}
    )
    
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("混淆矩阵", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n混淆矩阵已保存至: {save_path}")

# -------------------------- 8. 打印评估结果 --------------------------
def print_results(overall_metrics: dict, cm: np.ndarray, idx_to_class: dict):
    print("\n" + "="*80)
    print("                          测试集评估结果")
    print("="*80)
    
    print("\n【整体评估指标】 (论文 Sec 5 [cite: 359])")
    print(f"Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"F1-score (weighted):  {overall_metrics['f1-score']:.4f}")
    
    
    print("\n【混淆矩阵（文本格式）】")
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    print(" " * 15 + " ".join([f"{name[:8]:8s}" for name in class_names]))
    for i, (idx, name) in enumerate(sorted(idx_to_class.items())):
        row = cm[i]
        print(f"{name[:15]:15s}" + " ".join([f"{val:8d}" for val in row]))

# -------------------------- 9. 主函数 --------------------------
def main():
    args = parse_args()

    dataset_name = args.dataset_name
    if args.dataset_root:
        HPARAMS["dataset_root"] = args.dataset_root
    else:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        HPARAMS["dataset_root"] = os.path.join(project_dir, "data", dataset_name)

    if args.device_id is not None:
        HPARAMS["device_id"] = args.device_id
    device = get_device(HPARAMS["device_id"])

    # 1. 加载测试数据
    # 传递 feature 和 dataset_name
    test_loader, class_to_idx, idx_to_class = load_test_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"],
        feature=HPARAMS["feature"],
        dataset_name=dataset_name
    )
    HPARAMS["num_classes"] = len(class_to_idx)
    print(f"检测到类别数: {HPARAMS['num_classes']} (数据集: {dataset_name})")
    
    # 2. 初始化模型并加载权重
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else os.path.join(
        "./checkpoints", dataset_name, "best_val_model.pth"
    )

    model = init_model(
        num_classes=HPARAMS["num_classes"],
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # 3. 推理并收集标签
    y_true, y_pred = infer(model=model, test_loader=test_loader, device=device)
    
    # 4. 计算评估指标
    overall_metrics, cm = calculate_metrics(y_true, y_pred, idx_to_class)
    
    # 5. 打印结果
    print_results(overall_metrics, cm, idx_to_class)
    
    # 6. 绘制并保存混淆矩阵
    cm_save_path = os.path.join(HPARAMS["result_dir"], dataset_name, HPARAMS["cm_fig_name"])
    plot_confusion_matrix(cm, idx_to_class, cm_save_path)
    
    print("\n" + "="*80)
    print("测试完成！所有结果已输出，混淆矩阵已保存。")

if __name__ == "__main__":
    main()