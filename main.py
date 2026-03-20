# main.py
import argparse
import yaml
import os
import sys

# 打包后（PyInstaller）资源路径
def _get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

from scripts.train import train
from scripts.evaluate import evaluate
from scripts.infer import infer
from util.helper_functions import generate_labels, split_dataset


def load_config(config_path):
    """加载配置文件。打包运行时，将相对路径解析为相对于 config 所在目录。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Configuration file is empty or invalid.")
            # 解析相对路径为绝对路径（相对于 config 所在目录，便于从任意 cwd 运行）
            def resolve_path(base, path):
                if path and not os.path.isabs(path):
                    return os.path.normpath(os.path.join(base, path))
                return path

            if 'data' in config:
                for key in ('input_dirs', 'train_dir', 'train_labels', 'val_dir', 'val_labels'):
                    if key in config['data']:
                        config['data'][key] = resolve_path(config_dir, config['data'][key])
            if 'model' in config:
                if 'checkpoint_path' in config['model']:
                    config['model']['checkpoint_path'] = resolve_path(config_dir, config['model']['checkpoint_path'])
                if 'save_dir' in config['model']:
                    config['model']['save_dir'] = resolve_path(config_dir, config['model']['save_dir'])
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
    # 解析命令行参数（支持调试时覆盖）
    parser = argparse.ArgumentParser(description="虚拟人脸识别 / 视频真伪分类")
    parser.add_argument('--mode', choices=['load_videos', 'train', 'evaluate', 'infer', 'gui'], default='gui',
                        help="运行模式")
    default_config = os.path.join(_get_base_path(), 'config.yaml')
    parser.add_argument('--config', type=str, default=default_config, help="配置文件路径")
    parser.add_argument('--input', type=str, help="推理模式下的输入视频路径")
    args = parser.parse_args()
    mode = args.mode
    config_path = args.config
    input_video = args.input
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
        split_dataset(input_dirs, train_dir, val_dir, train_ratio=0.8)
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
            raise ValueError("Input video path is required for inference mode. Use --input <path>")
        print("Running in INFER mode")
        infer(config, input_video)
    elif mode == 'gui':
        from gui.app import run_gui
        run_gui(config)
