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
from torchvision.io import read_video
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
        # 获取视频文件路径和标签
        video_name = list(self.labels.keys())[idx]
        video_path = os.path.join(self.video_dir, video_name)
        label = self.labels[video_name]

        # 读取视频并抽取帧
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        video_frames = self._sample_frames(video_frames)

        # 调整维度为 (num_frames, C, H, W) 格式
        video_frames = video_frames.permute(0, 3, 1, 2)  # 将 (num_frames, H, W, C) 转换为 (num_frames, C, H, W)

        # 将每帧转换为 PIL Image 然后应用变换
        frames = [ToPILImage()(frame) for frame in video_frames]  # 转换为 PIL Image
        frames = [self.transform(frame) for frame in frames]  # 对每帧应用 transform
        frames = torch.stack(frames)  # shape: (num_frames, C, H, W)

        return {'video': frames, 'label': label}

    def _sample_frames(self, video_frames):
        """
        从视频帧中采样 num_frames 帧。
        如果视频帧数不足 num_frames，则重复最后一帧。
        """
        total_frames = video_frames.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()

        sampled_frames = video_frames[indices]
        if sampled_frames.shape[0] < self.num_frames:
            # 补充不足的帧
            pad = self.num_frames - sampled_frames.shape[0]
            last_frame = sampled_frames[-1].unsqueeze(0).repeat(pad, 1, 1, 1)
            sampled_frames = torch.cat([sampled_frames, last_frame], dim=0)

        return sampled_frames


def create_data_loader(video_dir, labels, batch_size=8, subset_ratio=0.1, num_frames=16, frame_size=128, shuffle=True):
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

    data_loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return data_loader

