import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Tuple
from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import CMoE, init_weights

# -------------------------- 1. 超参数--------------------------
HPARAMS = {
    # 设备配置
    "device_id": "0",  # 指定设备编号
    # 数据路径
    "dataset_root": "C:/Users/win11/Desktop/CMOE/data/shipsear9_5s",  # 可由命令行参数覆盖
    
    # 训练参数 (与论文 Sec 4.4 [cite: 349, 350, 351] 对齐)
    "batch_size": 16,
    "epochs": 60,                # 论文: 200 轮 
    "init_lr": 0.0005,            # 论文: 5e-4 
    "weight_decay": 1e-5,         # 论文: 1e-5 (AdamW) 
    
    # 学习率调度器
    "lr_decay_step": 20,
    "lr_decay_gamma": 0.2,
    
    # 模型参数 (与论文 Sec 3.3, Table 1 [cite: 182, 282] 对齐)
    "num_classes": 9,             # 与数据集实际类别数一致
    "residual": True,            # False=CMoE, True=RCMoE 
    "residual_scale": 1,       # 残差专家缩放系数
    
    # 特征选择：stft | mel | bark | cqt
    "feature": "mel",             # 论文测试了全部四种 [cite: 136]
    
    # CMoE配置 (与论文 Sec 3.3, 3.4 [cite: 213, 307] 对齐)
    "num_experts": 4,             # 论文测试了 2, 4, 8 [cite: 591]
    "gate_temperature": 1.0,
    
    # 损失项系数 (与论文 Sec 3.4 [cite: 307] 对齐)
    "lb_loss_weight": 0.01,       # 论文: alpha = 1e-2 [cite: 307]
    
    # 保存路径
    "checkpoint_dir": "./checkpoints",
    "best_ckpt_name": "best_val_model.pth",
    "final_ckpt_name": "final_epoch_model.pth"
}


def parse_args():
    parser = argparse.ArgumentParser(description="CMoE 训练入口")
    parser.add_argument("--dataset_name", type=str, default="shipsear9_5s",
                        choices=["shipsear9_5s", "oceanship_5s", "deepship_5s_id", "deepship_5s_normal"], help="选择数据集名称")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="数据集根路径（若不提供，将按项目目录下 data/{dataset_name} 推断）")
    parser.add_argument("--device_id", type=str, default=None, help="CUDA 设备编号，如 '0' 或 '0,1'")
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

# -------------------------- 3. 数据加载 --------------------------
def load_data(dataset_root: str, batch_size: int, feature: str, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """加载训练集和验证集"""
    # 训练集
    train_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=False,
        val_split_ratio=0.15,  # 论文 4.3 节: 15% 验证集 [cite: 342]
        random_seed=42
    )
    # 验证集
    val_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=True,
        val_split_ratio=0.15,
        random_seed=42
    )
    
    # 创建DataLoader
    # 传递 dataset_name 以便 feature_extraction 获取正确参数
    collate_fn = get_collate_fn(feature, dataset_name)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n数据集信息:")
    print(f"训练集样本数: {len(train_dataset)} (类别分布: {train_dataset.get_split_info()['class_distribution']})")
    print(f"验证集样本数: {len(val_dataset)} (类别分布: {val_dataset.get_split_info()['class_distribution']})")
    print(f"类别映射: {train_dataset.get_class_mapping()}")
    print(f"使用特征: {feature} (数据集: {dataset_name})")
    
    return train_loader, val_loader

# -------------------------- 4. 训练/验证核心函数 (与论文损失一致 [cite: 309]) --------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """训练单轮，返回训练集平均损失和准确率"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits, _, aux_losses = model(features)
        
        # 计算总损失 L = L_CE + alpha * L_balance [cite: 309]
        loss = criterion(logits, labels)
        if isinstance(aux_losses, dict) and "load_balance" in aux_losses:
            loss = loss + HPARAMS["lb_loss_weight"] * aux_losses["load_balance"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        pred_labels = torch.argmax(logits, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += features.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            batch_acc = (pred_labels == labels).sum().item() / features.size(0)
            print(f"  批次 {batch_idx+1:3d}/{len(loader)} | 批损失: {loss.item():.4f} | 批准确率: {batch_acc:.4f}")
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> Tuple[float, float]:
    """验证单轮，返回验证集平均损失和准确率"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits, _, aux_losses = model(features)
        
        # 计算总损失 L = L_CE + alpha * L_balance [cite: 309]
        loss = criterion(logits, labels)
        if isinstance(aux_losses, dict) and "load_balance" in aux_losses:
            loss = loss + HPARAMS["lb_loss_weight"] * aux_losses["load_balance"]
        
        total_loss += loss.item() * features.size(0)
        pred_labels = torch.argmax(logits, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# -------------------------- 5. Checkpoint保存函数 --------------------------
def save_checkpoint(save_path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler: StepLR, epoch: int, best_val_acc: float = None) -> None:
    """保存模型Checkpoint（包含模型、优化器、调度器状态）"""
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint已保存至: {save_path}")

# -------------------------- 6. 主训练流程--------------------------
def main():
    args = parse_args()

    # 解析数据路径
    dataset_name = args.dataset_name
    if args.dataset_root:
        HPARAMS["dataset_root"] = args.dataset_root
    else:
        project_dir = os.path.abspath(os.path.dirname(__file__))
        HPARAMS["dataset_root"] = os.path.join(project_dir, "data", dataset_name)

    # 设备
    if args.device_id is not None:
        HPARAMS["device_id"] = args.device_id
    device = get_device(HPARAMS["device_id"])
    
    # 1. 加载数据
    # 传递 dataset_name
    train_loader, val_loader = load_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"],
        feature=HPARAMS["feature"],
        dataset_name=dataset_name 
    )
    
    try:
        inferred_num_classes = len(train_loader.dataset.classes)
        HPARAMS["num_classes"] = inferred_num_classes
        print(f"检测到类别数: {HPARAMS['num_classes']} (数据集: {dataset_name})")
    except Exception:
        pass
    
    # 2. 初始化模型 (与论文 Table 1  一致)
    # 移除未使用的 input_dim, top_k, gate_hidden_dim
    model = CMoE(
        num_classes=HPARAMS["num_classes"],
        num_experts=HPARAMS["num_experts"],
        gate_temperature=HPARAMS["gate_temperature"],
        residual=HPARAMS["residual"],
        residual_scale=HPARAMS["residual_scale"]
    )
    print(f"使用CMoE模型 (Residual={HPARAMS['residual']})：K={HPARAMS['num_experts']}")
    model.apply(init_weights)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"启用多GPU训练: {torch.cuda.device_count()}张GPU")
    
    # 3. 初始化优化器、调度器、损失函数
    # 使用 AdamW 并添加 weight_decay [cite: 349, 350]
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=HPARAMS["init_lr"], 
        weight_decay=HPARAMS["weight_decay"]
    )
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=HPARAMS["lr_decay_step"],
        gamma=HPARAMS["lr_decay_gamma"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # 4. 记录最优模型
    best_val_acc = 0.0
    
    # 5. 开始训练循环
    print(f"\n=== 开始训练（共{HPARAMS['epochs']}轮）===")
    best_ckpt_path = "" # 初始化
    
    for epoch in range(1, HPARAMS["epochs"] + 1):
        print(f"\n【轮次 {epoch:3d}/{HPARAMS['epochs']}】")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        print(f"轮次总结:")
        print(f"  训练集 - 损失: {train_loss:.4f} | 准确率: {train_acc:.4f}")
        print(f"  验证集 - 损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
        
        # 6. 更新最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  发现新的最优验证集准确率: {best_val_acc:.4f}")
            best_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], dataset_name, HPARAMS["best_ckpt_name"])
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc
            )
        
        # 7. 更新学习率
        scheduler.step()
    
    # 8. 训练结束后保存最后一轮模型
    final_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], dataset_name, HPARAMS["final_ckpt_name"])
    save_checkpoint(
        save_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=HPARAMS["epochs"],
        best_val_acc=best_val_acc # 记录下训练结束时的最优val
    )
    
    # 9. 训练总结
    print(f"\n=== 训练完成 ===")
    print(f"最优验证集准确率: {best_val_acc:.4f}")
    if best_ckpt_path:
        print(f"最优模型路径: {os.path.abspath(best_ckpt_path)}")
    print(f"最后一轮模型路径: {os.path.abspath(final_ckpt_path)}")


if __name__ == "__main__":
    main()