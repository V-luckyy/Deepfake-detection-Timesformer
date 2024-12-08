# main.py
import argparse
import yaml
import os
from scripts.train import train
from scripts.evaluate import evaluate
from scripts.infer import infer
from util.helper_functions import generate_labels, split_data


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Configuration file is empty or invalid.")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")


if __name__ == '__main__':
    # 检查是否在调试模式下运行
    # if os.getenv('PYTHON_DEBUG_MODE', '0') == '1':
    #     # 默认设置调试参数
    #     mode = "load_videos"
    #     config_path = "config.yaml"
    #     input_video = None
    # else:
    #     # 解析命令行参数
    #     parser = argparse.ArgumentParser(description="Video classification project")
    #     parser.add_argument('--mode', choices=['load_videos', 'train', 'evaluate', 'infer'], required=True,
    #                         help="Specify the mode: load_videos, train, evaluate, or infer")
    #     parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    #     parser.add_argument('--input', type=str, help="Path to input video for inference (required for infer mode)")
    #     args = parser.parse_args()
    #
    #     mode = args.mode
    #     config_path = args.config
    #     input_video = args.input
    mode = "infer"
    config_path = "config.yaml"
    input_video = "data/Celeb-DF-V2/id0_id1_00001.mp4"
    # 加载配置文件
    config = load_config(config_path)
    input_dirs = config['data']['input_dirs']
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    train_label_path = config['data']['train_labels']
    val_label_path = config['data']['val_labels']

    # 根据 mode 执行对应功能
    if mode == 'load_videos':
        print("Running in LOAD_VIDEOS mode")
        split_data(input_dirs, train_dir, val_dir)
        generate_labels(train_dir, train_label_path)
        generate_labels(val_dir, val_label_path)
    elif mode == 'train':
        print("Running in TRAIN mode")
        train(config)
    elif mode == 'evaluate':
        print("Running in EVALUATE mode")
        evaluate(config)
    elif mode == 'infer':
        if not input_video:
            raise ValueError("Input video path is required for inference mode.")
        print("Running in INFER mode")
        infer(config, input_video)
