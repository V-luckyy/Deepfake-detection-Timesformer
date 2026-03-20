"""VideoDataset 类：用于加载和预处理单个视频数据。

__init__：初始化数据集对象，设置视频路径、标签映射、帧数、帧尺寸和数据增强操作。
__getitem__：根据索引 idx 获取视频和标签，对视频帧进行采样和转换，然后返回视频张量和标签。
_sample_frames：从视频中抽取指定数量的帧。如果视频帧数不足，则重复最后一帧，确保帧数一致。
create_data_loader 函数：创建 DataLoader，便于批量加载数据。

参数说明：
video_dir：视频存放路径。
labels：标签字典。
batch_size：训练批次大小。
num_frames 和 frame_size：用于指定每个视频的帧数和帧分辨率。
shuffle：是否打乱数据集顺序。
返回值：返回 DataLoader 对象，方便后续训练和验证时批量加载数据。"""
import json

# data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import cv2
import os
import random

from torchvision.transforms import ToPILImage


class VideoDataset(Dataset):
    def __init__(self, video_dir, labels, num_frames=16, frame_size=128, transform=None):
        """
        Args:
            video_dir (str): 存放视频文件的路径。
            labels (dict): 字典格式的标签映射，键为视频文件名，值为标签（如真实/虚假标签）。
            num_frames (int): 处理后的每个视频的帧数。
            frame_size (int): 每帧的分辨率（高度和宽度）。
            transform (torchvision.transforms): 图像数据增强（可选）。
        """
        self.video_dir = video_dir
        # 自动加载 JSON 文件为字典
        if isinstance(labels, str):
            with open(labels, 'r') as f:
                self.labels = json.load(f)  # 加载为字典
        elif isinstance(labels, dict):
            self.labels = labels
        else:
            raise ValueError("labels must be a dict or a path to a JSON file.")
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.frame_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_name = list(self.labels.keys())[idx]
        video_path = os.path.join(self.video_dir, video_name)
        label = self.labels[video_name]

        # 使用 cv2 逐帧读取，避免整段视频加载导致内存溢出
        frames = self._load_frames_cv2(video_path)
        return {'video': frames, 'label': label}

    def _load_frames_cv2(self, video_path):
        """用 cv2.VideoCapture 按需读取帧，仅保留采样帧，降低内存占用。"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"无法读取视频帧数: {video_path}")

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
        frames = []
        last_frame = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    raise ValueError(f"无法读取视频帧: {video_path}")
            else:
                last_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ToPILImage()(frame)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()
        return torch.stack(frames)


def create_data_loader(video_dir, labels, batch_size=8, subset_ratio=0.1, num_frames=16, frame_size=128, shuffle=True, num_workers=0):
    """
    创建用于训练的 DataLoader。

    Args:
        video_dir (str): 视频存放路径。
        labels (dict): 标签映射。
        batch_size (int): 批次大小。
        num_frames (int): 目标帧数。
        frame_size (int): 每帧的分辨率。
        shuffle (bool): 是否打乱数据。

    Returns:
        DataLoader: 用于训练的视频 DataLoader。
    """
    dataset = VideoDataset(
        video_dir=video_dir,
        labels=labels,
        num_frames=num_frames,
        frame_size=frame_size,
        transform=transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    # 计算选择的样本数量
    subset_size = int(len(dataset) * subset_ratio)
    # 随机选择 subset_size 个样本
    subset_indices = torch.randperm(len(dataset)).tolist()[:subset_size]
    subset = Subset(dataset, subset_indices)

    data_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

