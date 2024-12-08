"""project_root/
├── data/                   # 数据集存放路径
│   ├── train_videos            # 训练视频数据集
│   └── val_videos           # 验证视频数据集
├── models/                 # 存放模型定义的文件夹
│   ├── model.py            # 视频 Transformer 模型定义                   √
│   └── layers.py           # 自定义模型层（如位置编码、自注意力层）           √
├── scripts/
│   ├── train.py            # 训练脚本                                  √
│   ├── evaluate.py         # 模型评估脚本                               √
│   └── infer.py            # 推理脚本                                  √
├── util/
│   ├── data_loader.py      # 数据加载与预处理                          √
│   ├── metrics.py          # 评估指标（如准确率、F1分数、AUC等）          √
│   ├── helper_functions.py # 其他辅助函数                             √
├── config.yaml             # 配置文件（包含超参数和路径配置）             √
└── main.py                 # 主程序入口，整合各个模块"""

# 训练模型
# python main.py --mode train --config config.yaml
# 评估模型
# python main.py --mode evaluate --config config.yaml
# 推理/实时检测
# python main.py --mode infer --config config.yaml --input video.mp4
