# scripts/infer.py
import torch
import cv2
from models.model import VideoTransformer
from util.helper_functions import preprocess_video
import os


def infer(config, video_path):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # 加载模型检查点
    checkpoint_path = config['model'].get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        raise FileNotFoundError("Model checkpoint path not found. "
                                "Please provide a valid checkpoint path in config.yaml.")

    # 读取并预处理视频文件
    frames = preprocess_video(video_path, config['data']['num_frames'], config['data']['frame_size'])
    frames = frames.to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(frames.unsqueeze(0))  # 添加批次维度
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()

    # 输出推理结果
    labels = config.get('labels', {0: "Real", 1: "Fake"})
    print(f"Inference Result: {labels[pred_class]} (Confidence: {confidence:.4f})")
