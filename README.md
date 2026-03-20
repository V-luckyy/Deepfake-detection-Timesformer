# 虚拟人脸识别 - Video Deepfake Detection

基于视频的人脸真伪检测（Real / Fake）GUI，支持推理检测与训练。

---

## 一、运行 GUI

在项目根目录下执行：

```bash
pip install -r requirements.txt
python main.py
```

或指定模式与配置：

```bash
python main.py --mode gui --config config.yaml
```

---

## 二、程序文件结构（测试 GUI 所需）

```
face_recognition/
├── main.py                 # 入口，默认启动 GUI
├── config.yaml             # 配置文件（数据/模型路径、超参）
├── requirements.txt        # Python 依赖
│
├── gui/
│   ├── __init__.py
│   └── app.py              # GUI 主界面（推理 + 训练）
│
├── scripts/
│   ├── infer.py            # 推理逻辑
│   ├── train.py            # 训练逻辑
│   └── evaluate.py         # 评估逻辑
│
├── models/
│   ├── model.py            # VideoTransformer 模型
│   ├── layers.py           # 模型层
│   └── checkpoints/       # 存放训练得到的 .pth 权重（推理时可选）
│        └── *.pth
│
├── util/
│   ├── helper_functions.py # 预处理、标签生成等
│   ├── data_loader.py      # 视频数据集与 DataLoader
│   └── metrics.py         # 评估指标
│
└── data/                   # 见下文「数据与视频」
    ├── train_labels.json
    ├── val_labels.json
    ├── train_videos/      # 训练视频（文件名需与 train_labels 键一致）
    └── val_videos/        # 验证视频（文件名需与 val_labels 键一致）
```

说明：仅列出运行与测试 GUI 所需的程序与数据；其他脚本、材料、第三方子项目等未列出。

---

## 三、数据与视频文件

### 1. 推理 / 检测

- **模型**：至少一个 `.pth` 权重文件。  
  - 可放在 `models/checkpoints/`，启动后会在下拉框列出；  
  - 或通过「浏览...」选择任意路径的 `.pth`。
- **视频**：任意一个视频文件（如 `.mp4`、`.avi`、`.mov`、`.mkv`），用于检测真伪。

### 2. 训练

需在 `config.yaml` 中配置（或保持默认），并保证路径与内容一致：

| 配置项 | 含义 | 要求 |
|--------|------|------|
| `data.train_dir` | 训练视频目录 | 目录存在，其下为视频文件 |
| `data.val_dir` | 验证视频目录 | 同上 |
| `data.train_labels` | 训练标签 JSON 路径 | 见下 |
| `data.val_labels` | 验证标签 JSON 路径 | 见下 |

**标签 JSON 格式**（键为视频文件名，值为 0/1）：

```json
{
  "00000.mp4": 0,
  "00001.mp4": 1,
  ...
}
```

- `0`：Real（真实）
- `1`：Fake（伪造）

**目录与标签对应关系**：  
`train_labels.json` 的键必须与 `train_dir` 下文件名一致；`val_labels.json` 与 `val_dir` 同理。  
例如：若键为 `"00000.mp4"`，则需存在 `train_videos/00000.mp4`（或 `val_videos/00000.mp4`）。

### 3. 默认路径（config.yaml）

- 训练/验证数据：`data/train_videos`、`data/val_videos`
- 标签文件：`data/train_labels.json`、`data/val_labels.json`
- 模型保存/加载：`models/checkpoints/`，推理时会扫描该目录下的 `.pth`

程序会按「相对于 config 所在目录」解析这些路径，因此从任意当前目录运行 `python main.py` 均可，只要项目结构如上即可。

---

## 四、GUI 功能简述

- **推理 / 检测**：选择模型文件（.pth）和视频文件，点击「开始检测」，查看 Real/Fake 及置信度。
- **训练**：在界面填写或选择「模型保存位置」，点击「开始训练」；可查看进度条、日志、最终验证准确率及模型保存路径。训练数据路径以 `config.yaml` 为准。
