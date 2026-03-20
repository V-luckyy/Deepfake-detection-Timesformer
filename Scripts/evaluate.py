# scripts/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model import VideoTransformer
from util.data_loader import create_data_loader
from util.metrics import calculate_metrics
import os


def evaluate(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载验证数据集
    val_loader = create_data_loader(
        video_dir=config['data']['val_dir'],
        labels=config['data']['val_labels'],
        batch_size=config['evaluation']['batch_size'],
        subset_ratio=config['evaluation'].get('subset_ratio', 1.0),
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        shuffle=False
    )

    # 初始化并加载模型
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

    # 加载训练好的模型权重
    checkpoint_path = config['model'].get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        raise FileNotFoundError("Model checkpoint path not found. Please provide a valid checkpoint path in config.yaml.")

    # 计算评估指标
    model.eval()
    with torch.no_grad():
        metrics = calculate_metrics(model, val_loader, device=device)

    # 输出评估结果
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
