# train.py
import torch
import torch.optim as optim
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model import VideoTransformer
from util.data_loader import create_data_loader
from util.metrics import calculate_metrics
from util.helper_functions import plot_training_progress
import os


def train(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建训练集数据加载器
    train_loader = create_data_loader(
        video_dir=config['data']['train_dir'],
        labels=config['data']['train_labels'],
        batch_size=config['training']['batch_size'],
        subset_ratio=config['training']['subset_ratio'],
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        shuffle=True  # 训练集需要打乱数据
    )

    # 创建验证集数据加载器
    val_loader = create_data_loader(
        video_dir=config['data']['val_dir'],
        labels=config['data']['val_labels'],
        batch_size=config['evaluation']['batch_size'],
        subset_ratio=config['evaluation']['subset_ratio'],
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        shuffle=False  # 验证集通常不打乱
    )

    # 初始化模型并移动到设备
    model = VideoTransformer(
        num_frames=config['model']['num_frames'],
        num_patches=config['model']['num_patches'],
        frame_size=config['data']['frame_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        num_classes=config['model']['num_classes']
    ).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=config['training']['patience'])

    # 初始化记录指标的列表
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # 训练循环
    num_epochs = config['training']['epochs']
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            # 数据移动到设备
            inputs = batch['video'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

        # 计算并打印每个 epoch 的训练损失和准确率
        epoch_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_metrics = calculate_metrics(model, val_loader, criterion, device=device)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            print(f"Validation Metrics at Epoch {epoch + 1}: {val_metrics}")

        # 根据验证集损失调整学习率
        scheduler.step(val_metrics['loss'])

        # 保存模型检查点
        if (epoch+1) % config['training']['checkpoint_interval'] == 0:
            if not os.path.exists(config['model']['save_dir']):
                os.makedirs(config['model']['save_dir'])
            checkpoint_path = os.path.join(config['model']['save_dir'], f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
    # 绘制训练和验证过程的图表
    plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies)
