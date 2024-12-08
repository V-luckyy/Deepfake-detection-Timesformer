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


# 预处理视频
def preprocess_video(video_path, num_frames, frame_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor()
    ])

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


# 数据增强
def augment_video(video_path, output_dir, augment_count=10):
    """
    对视频进行数据增强，保存增强后的多个副本。

    video_path: 原始视频路径
    output_dir: 增强后的视频保存目录
    augment_count: 生成的增强视频数量（默认为 10）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # 数据增强操作
    for i in range(augment_count):
        augmented_frames = frames.copy()

        # 随机选择一种数据增强方式（旋转，翻转，模糊，裁剪等）
        if random.random() > 0.5:
            augmented_frames = [cv2.flip(frame, 1) for frame in augmented_frames]  # 水平翻转
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            augmented_frames = [cv2.rotate(frame, angle) for frame in augmented_frames]  # 随机旋转

        # 保存增强的视频
        augmented_video_path = os.path.join(output_dir,
                                            f"{os.path.basename(video_path).split('.')[0]}_augmented_{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(augmented_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (frames[0].shape[1], frames[0].shape[0]))

        for frame in augmented_frames:
            out.write(frame)

        out.release()
        # print(f"增强视频已保存至: {augmented_video_path}")


# def split_and_augment_videos(input_dirs, output_train_dir, output_val_dir, train_ratio=0.8):
#     """
#     将视频按 5:1 的比例分成训练集和验证集，确保真假视频样本均衡。
#     当真实视频数量不够时，调用数据增强函数增加真实视频样本。
#
#     input_dirs: 包含视频的文件夹路径字典，例如：
#                 {'real': 'path_to_Celeb-real', 'synthesis': 'path_to_Celeb-synthesis',
#                 'youtube_real': 'path_to_YouTube-real'}
#     output_train_dir: 训练集存放路径
#     output_val_dir: 验证集存放路径
#     train_ratio: 训练集与验证集的比例（默认 0.8）
#     augment_count: 增强视频的数量（默认为 10）
#     """
#     # 创建训练集和验证集的目标文件夹
#     os.makedirs(output_train_dir, exist_ok=True)
#     os.makedirs(output_val_dir, exist_ok=True)
#
#     # 真实和合成的视频路径
#     real_videos = [os.path.join(input_dirs['real'], f) for f in os.listdir(input_dirs['real']) if f.endswith('.mp4')]
#     synthesis_videos = [os.path.join(input_dirs['synthesis'], f) for f in os.listdir(input_dirs['synthesis']) if
#                         f.endswith('.mp4')]
#     real_videos = real_videos + [os.path.join(input_dirs['youtube_real'], f)
#                                  for f in os.listdir(input_dirs['youtube_real']) if f.endswith('.mp4')]
#
#     # 打乱视频
#     random.shuffle(real_videos)
#     random.shuffle(synthesis_videos)
#
#     # 真实视频与合成视频比例，确保训练集和验证集均衡
#     train_real, val_real = train_test_split(real_videos, train_size=train_ratio, random_state=42)
#     train_synthesis, val_synthesis = train_test_split(synthesis_videos, train_size=train_ratio, random_state=42)
#
#     # 将文件复制到目标文件夹
#     def copy_videos(videos, dest_dir):
#         for video in videos:
#             shutil.copy(video, dest_dir)
#
#     # 拷贝到训练集和验证集
#     copy_videos(train_synthesis, output_train_dir)
#     copy_videos(val_synthesis, output_val_dir)
#     copy_videos(train_real, output_train_dir)
#     copy_videos(val_real, output_val_dir)
#
#     # 当真实视频样本数不足时，进行数据增强
#     if len(train_real) < len(train_synthesis):
#         print(f"训练集中的真实视频不足，正在进行数据增强...")
#         augment_count = (len(train_synthesis) - len(train_real)) // len(train_real)
#         augment_video_list(train_real, output_train_dir, augment_count)
#         augment_video_list(val_real, output_val_dir, augment_count)
#     else:
#         copy_videos(train_real, output_train_dir)
#         copy_videos(val_real, output_val_dir)
#
#     print(f"训练集和验证集已分配到 {output_train_dir} 和 {output_val_dir}，分别包含真实和合成的视频。")

def split_data(src_dir, train_dir, val_dir, train_ratio=0.8):
    """
    划分数据集到训练集和验证集，保持真实和合成数据的比例平衡。

    参数:
    - src_dir: 数据集根目录 (Celeb-DF-v2)
    - train_dir: 训练集目标目录
    - val_dir: 验证集目标目录
    - train_ratio: 训练集占比，默认为0.8（80%训练集，20%验证集）
    """
    # 获取真实和合成数据的路径
    real_dir = os.path.join(src_dir, 'Celeb-real')
    synthesis_dir = os.path.join(src_dir, 'Celeb-synthesis')

    # 获取各自文件夹下的视频文件名
    real_videos = os.listdir(real_dir)
    synthesis_videos = os.listdir(synthesis_dir)

    # 打乱视频文件顺序
    random.shuffle(real_videos)
    random.shuffle(synthesis_videos)

    # 计算划分的数量
    real_train_count = int(len(real_videos) * train_ratio)
    synthesis_train_count = int(len(synthesis_videos) * train_ratio)

    # 确保训练集和验证集的比例是平衡的
    real_val_count = len(real_videos) - real_train_count
    synthesis_val_count = len(synthesis_videos) - synthesis_train_count

    # 创建目标目录（训练集和验证集）
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 训练集
    real_train = real_videos[:real_train_count]
    synthesis_train = synthesis_videos[:synthesis_train_count]

    # 验证集
    real_val = real_videos[real_train_count:]
    synthesis_val = synthesis_videos[synthesis_train_count:]

    # 将文件从源目录复制到目标目录
    def copy_files(files, src_folder, target_folder):
        for file in files:
            shutil.copy(os.path.join(src_folder, file), target_folder)

    # 将真实和合成数据分别复制到训练集和验证集目录
    copy_files(real_train, real_dir, os.path.join(train_dir, 'Celeb-real'))
    copy_files(synthesis_train, synthesis_dir, os.path.join(train_dir, 'Celeb-synthesis'))

    copy_files(real_val, real_dir, os.path.join(val_dir, 'Celeb-real'))
    copy_files(synthesis_val, synthesis_dir, os.path.join(val_dir, 'Celeb-synthesis'))


def augment_video_list(video_list, output_dir, augment_count):
    """
    对视频列表进行数据增强并将增强数据保存至指定路径。

    video_list: 需要增强的视频文件路径列表
    output_dir: 增强数据保存路径
    augment_count: 每个视频生成的增强样本数
    split_type: 'train' 或 'val'，分别表示训练集或验证集
    """
    for video_path in video_list:
        augment_video(video_path, output_dir, augment_count)
        print(f"增强视频添加至: {output_dir}")


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
