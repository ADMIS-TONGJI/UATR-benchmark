import os
import torch
HPARAMS_DEVICE = {
    "device_id": "4"  
}

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = HPARAMS_DEVICE["device_id"]
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Tuple
from dataset import UnderwaterAudioDataset
from multi_feature import collate_fn  
from model import CNN_LSTM_Model  


# -------------------------- 1. 超参数配置 --------------------------
HPARAMS = {
    # 设备配置
    "device_id": HPARAMS_DEVICE["device_id"],  # 指定GPU设备编号
    # 数据路径
    "dataset_root": "./deepship",  # 数据集根目录
    # 训练参数
    "batch_size": 32,
    "epochs": 100,
    "init_lr": 0.0005,
    "lr_decay_step": 20,  # 每20轮衰减学习率
    "lr_decay_gamma": 0.4,  # 衰减系数
    # 早停机制
    "early_stop_patience": 20,  # 连续n轮验证集无提升则早停
    "num_classes": 4,  # 数据集实际类别数
    "checkpoint_dir": "./checkpoints",
    "best_ckpt_name": "best_val_model.pth",
    "final_ckpt_name": "final_epoch_model.pth"
}


# -------------------------- 2. 设备选择 --------------------------
def get_device(device_id: str = None) -> torch.device:
    """根据设备编号返回设备对象"""
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"使用GPU设备: {device_id} (共{torch.cuda.device_count()}张GPU)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备（训练效率较低，建议使用GPU）")
    return device

DEVICE = get_device(HPARAMS["device_id"])


# -------------------------- 3. 数据加载 --------------------------
def load_data(dataset_root: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """加载训练集和验证集"""
    train_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=False,
        val_split_ratio=0.15,  # 训练集:验证集 = 85:15
        random_seed=42  
    )

    val_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="train",
        is_validation=True,
        val_split_ratio=0.15,
        random_seed=42  
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        collate_fn=collate_fn,  
        num_workers=4, 
        pin_memory=True  
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 打印数据统计信息
    print(f"\n=== 数据集信息 ===")
    train_info = train_dataset.get_split_info()
    val_info = val_dataset.get_split_info()
    print(f"训练集 - 样本数: {len(train_dataset)} | 类别分布: {train_info['class_distribution']}")
    print(f"验证集 - 样本数: {len(val_dataset)} | 类别分布: {val_info['class_distribution']}")
    print(f"类别映射（名称→数字标签）: {train_dataset.get_class_mapping()}")
    
    # 打印融合特征维度信息
    if len(train_dataset) > 0:
        sample_path, _ = train_dataset[0]
        sample_feature = collate_fn([(sample_path, 0)])[0][0]
        print(f"融合特征维度: {sample_feature.shape}")  # 应输出(192, 1)
    
    return train_loader, val_loader


# -------------------------- 4. 训练/验证核心函数 --------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """训练单轮，返回训练集平均损失和准确率"""
    model.train()  # 模型切换至训练模式（启用Dropout、BN更新）
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(loader):
        # 数据移至目标设备并调整维度
        features = features.to(device, non_blocking=True).permute(0, 2, 1)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        logits, _ = model(features)
        loss = criterion(logits, labels)  
        
        # 反向传播+参数更新
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        
        # 统计训练指标
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size  
        pred_labels = torch.argmax(logits, dim=1)  
        total_correct += (pred_labels == labels).sum().item()  
        total_samples += batch_size
        
        # 打印批次进度
        if (batch_idx + 1) % 10 == 0:
            batch_acc = (pred_labels == labels).sum().item() / batch_size
            print(f"  批次 {batch_idx+1:3d}/{len(loader)} | 批损失: {loss.item():.4f} | 批准确率: {batch_acc:.4f}")
    
    # 计算单轮平均指标
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()  # 禁用梯度计算，加速验证并避免内存占用
def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> Tuple[float, float]:
    """验证单轮，返回验证集平均损失和准确率"""
    model.eval()  # 模型切换至评估模式（关闭Dropout、固定BN）
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for features, labels in loader:
        # 数据移至目标设备并调整维度
        features = features.to(device, non_blocking=True).permute(0, 2, 1)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        logits, _ = model(features)
        loss = criterion(logits, labels)
        
        # 统计验证指标
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        pred_labels = torch.argmax(logits, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += batch_size
    
    # 计算单轮平均指标
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# -------------------------- 5. Checkpoint保存函数 --------------------------
def save_checkpoint(save_path: str, model: nn.Module, optimizer: optim.Optimizer,
                    scheduler: StepLR, epoch: int, best_val_acc: float = None) -> None:
    """保存模型Checkpoint（含模型参数、优化器状态、训练进度）"""
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    # 封装Checkpoint内容
    checkpoint = {
        "epoch": epoch,  # 当前训练轮次
        "model_state_dict": model_state_dict,  # 模型参数
        "optimizer_state_dict": optimizer.state_dict(),  
        "scheduler_state_dict": scheduler.state_dict(),  
        "best_val_acc": best_val_acc,  
        "num_classes": HPARAMS["num_classes"]  
    }
    
    # 创建保存目录（若不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存Checkpoint
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint已保存至: {os.path.abspath(save_path)}")


# -------------------------- 6. 主训练流程 --------------------------
def main():
    # 1. 加载数据（训练集+验证集）
    print("=== 开始加载数据 ===")
    train_loader, val_loader = load_data(
        dataset_root=HPARAMS["dataset_root"],
        batch_size=HPARAMS["batch_size"]
    )
    
    # 2. 初始化CNN-LSTM模型
    print("\n=== 初始化CNN-LSTM模型 ===")
    model = CNN_LSTM_Model(
        num_classes=HPARAMS["num_classes"]  # 匹配数据集类别数
    )
    
    # 将模型移至目标设备
    model.to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"启用多GPU训练: 共{torch.cuda.device_count()}张GPU")
    
    # 3. 初始化训练组件（优化器、调度器、损失函数）
    print("\n=== 初始化训练组件 ===")
    optimizer = optim.Adam(
        model.parameters(),
        lr=HPARAMS["init_lr"],
        weight_decay=1e-5  
    )
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=HPARAMS["lr_decay_step"],
        gamma=HPARAMS["lr_decay_gamma"],
        verbose=True  # 打印学习率更新信息
    )
    criterion = nn.CrossEntropyLoss()
    
    # 4. 早停机制变量
    best_val_acc = 0.0  # 历史最高验证准确率
    prev_val_acc = 0.0  # 上一轮验证准确率
    no_improve_epochs = 0  # 连续无提升的轮次
    
    # 5. 开始训练循环
    print(f"\n=== 开始训练（共{HPARAMS['epochs']}轮）===")
    for epoch in range(1, HPARAMS["epochs"] + 1):
        print(f"\n【轮次 {epoch:2d}/{HPARAMS['epochs']}】")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练单轮
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )
        
        # 验证单轮
        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        
        # 打印轮次结果
        print(f"轮次总结:")
        print(f"  训练集 - 损失: {train_loss:.4f} | 准确率: {train_acc:.4f}")
        print(f"  验证集 - 损失: {val_loss:.4f} | 准确率: {val_acc:.4f}")
        
        if epoch == 1:
            # 第一轮无“上一轮”可比，直接初始化prev_val_acc，不累计无提升轮次
            prev_val_acc = val_acc
            print(f"  第一轮验证完成，初始化上一轮准确率为: {prev_val_acc:.4f}")
        else:
            # 从第二轮开始，与上一轮验证准确率对比
            if val_acc < prev_val_acc:
                no_improve_epochs += 1
                print(f"  验证集准确率低于上一轮（{val_acc:.4f} < {prev_val_acc:.4f}），连续无提升轮次: {no_improve_epochs}")
            else:
                # 准确率≥上一轮，重置无提升轮次
                no_improve_epochs = 0
                print(f"  验证集准确率≥上一轮（{val_acc:.4f} ≥ {prev_val_acc:.4f}），重置无提升轮次为0")
            # 更新“上一轮准确率”为当前轮，供下一轮对比
            prev_val_acc = val_acc

        # 6. 更新最优模型与Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存验证集最优模型
            best_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], HPARAMS["best_ckpt_name"])
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc
            )
        
        # 7. 早停判断
        if no_improve_epochs >= HPARAMS["early_stop_patience"]:
            print(f"\n早停触发！连续{HPARAMS['early_stop_patience']}轮验证集准确率无提升")
            break
        
        # 8. 更新学习率
        scheduler.step()
    
    # 9. 保存最后一轮模型
    final_epoch = epoch if no_improve_epochs < HPARAMS["early_stop_patience"] else HPARAMS["epochs"]
    final_ckpt_path = os.path.join(HPARAMS["checkpoint_dir"], HPARAMS["final_ckpt_name"])
    save_checkpoint(
        save_path=final_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epoch,
        best_val_acc=best_val_acc
    )
    
    # 10. 训练完成总结
    print(f"\n=== 训练流程全部完成 ===")
    print(f"关键结果：")
    print(f"  - 最优验证集准确率: {best_val_acc:.4f}")
    print(f"  - 最优模型路径: {os.path.abspath(best_ckpt_path)}")
    print(f"  - 最后一轮模型路径: {os.path.abspath(final_ckpt_path)}")
    print(f"  - 实际训练轮次: {final_epoch}")


if __name__ == "__main__":
    main()