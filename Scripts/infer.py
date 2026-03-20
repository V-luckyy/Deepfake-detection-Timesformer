# scripts/infer.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
from models.model import VideoTransformer
from util.helper_functions import preprocess_video
import os


def infer(config, video_path, checkpoint_path=None):
    """
    对视频进行真伪推理。
    :param config: 配置字典
    :param video_path: 视频文件路径
    :param checkpoint_path: 可选，覆盖 config 中的 checkpoint_path
    :return: (label_str, confidence) 如 ("Real", 0.95) 或 ("Fake", 0.87)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = checkpoint_path or config['model'].get('checkpoint_path', None)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")

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

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

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

    label_map = config.get('labels', {0: "Real", 1: "Fake"})
    label_str = label_map.get(pred_class, ["Real", "Fake"][pred_class])
    return label_str, confidence
