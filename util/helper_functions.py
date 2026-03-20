# util/helper_functions.py

import torch
import os
import cv2
import random
import shutil
import re
import json

from matplotlib import pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split


# 预处理视频（与训练时保持一致的归一化）
def preprocess_video(video_path, num_frames, frame_size, normalize=True):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor()
    ]
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    transform = transforms.Compose(transform_list)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames)  # Shape: (num_frames, channels, height, width)


# 生成标签文件
def generate_labels(video_dir, label_output_path):
    """
    根据视频文件的命名规则生成标签文件。

    参数:
    - video_dir: 视频文件夹路径
    - label_output_path: 输出标签 JSON 文件路径

    标签规则:
    - 如果文件名以 "idX_idY" 的模式开头，则标记为 1（虚假视频）
    - 否则标记为 0（真实视频）
    """
    labels = {}
    pattern = re.compile(r'^id\d+_id\d+_')  # 匹配以两个 id 开头的文件名

    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            # 使用正则表达式判断是否为虚假视频
            if pattern.match(video_file):
                labels[video_file] = 1  # 虚假视频
            else:
                labels[video_file] = 0  # 真实视频

    # 将标签写入 JSON 文件
    with open(label_output_path, 'w') as json_file:
        json.dump(labels, json_file, indent=4)
    print(f"Labels saved to {label_output_path}")


def split_dataset(source_dir, train_dir, val_dir, train_ratio=0.83):
    """
    将Celeb-real和Celeb-synthesis数据集按照5:1的比例划分为训练集和验证集，
    且确保每个数据集中的真实与合成人脸视频比例相同。

    :param source_dir: 数据集源目录 (Celeb-DF-v2)
    :param train_dir: 训练集保存的目标目录
    :param val_dir: 验证集保存的目标目录
    :param train_ratio: 训练集比例，默认为0.83 (5:1)
    """
    # 获取源数据集中的文件夹路径
    real_folder = os.path.join(source_dir, 'Celeb-real')
    synthesis_folder = os.path.join(source_dir, 'Celeb-synthesis')

    # 获取所有视频文件路径
    real_videos = [os.path.join(real_folder, f) for f in os.listdir(real_folder)]
    synthesis_videos = [os.path.join(synthesis_folder, f) for f in random.sample(os.listdir(synthesis_folder), 890)]

    # 打乱列表，确保数据划分的随机性
    random.shuffle(real_videos)
    random.shuffle(synthesis_videos)

    # 计算划分点
    real_train, real_val = train_test_split(real_videos, train_size=train_ratio, random_state=42)
    synthesis_train, synthesis_val = train_test_split(synthesis_videos, train_size=train_ratio, random_state=42)

    # 确保目标文件夹存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 创建训练集和验证集的文件夹
    os.makedirs(os.path.join(train_dir, 'Celeb-real'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Celeb-synthesis'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Celeb-real'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'Celeb-synthesis'), exist_ok=True)

    # 将文件移动到相应的目录
    def move_files(file_paths, target_folder):
        for file in file_paths:
            shutil.copy(file, target_folder)

    # 移动训练集和验证集数据
    move_files(real_train, os.path.join(train_dir, 'Celeb-real'))
    move_files(synthesis_train, os.path.join(train_dir, 'Celeb-synthesis'))
    move_files(real_val, os.path.join(val_dir, 'Celeb-real'))
    move_files(synthesis_val, os.path.join(val_dir, 'Celeb-synthesis'))

    print(f"训练集和验证集划分完成：")
    print(f"训练集 - 真实人脸视频: {len(real_train)}, 合成人脸视频: {len(synthesis_train)}")
    print(f"验证集 - 真实人脸视频: {len(real_val)}, 合成人脸视频: {len(synthesis_val)}")


# 绘制训练过程
def plot_training_progress(train_losses, train_accuracies, val_losses, val_accuracies):
    # 设置字体 Times New Roman
    plt.rc('font', family='Times New Roman')

    # 绘制训练损失和验证损失
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy", color="blue")
    plt.plot(val_accuracies, label="Validation Accuracy", color="red")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig('results.png', dpi=600)
    plt.show()
