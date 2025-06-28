# YOLO-MIA
## Membership Inference Attacks on YOLOv8 Object Detection Models
## 黑盒条件下的基于YOLOv8目标检测模型的成员推理攻击

## 各脚本作用
- `train.py`：训练YOLOv8n目标检测模型，同时也是我们要进行攻击的模型
- `train_base`：原来准备的全知模型训练脚本，时间紧迫没有选择使用参数最大的YOLOv8x模型
- `train_v8l`：实际使用的全知模型训练脚本，使用YOLOv8l模型
- `split.py`：用于抽取VOC数据集中的一部分作为训练集和验证集，用于训练被攻击模型并为攻击模型提供训练数据
- `get_data.py`：调用v8n和v8l模型，对split.py抽取的训练集和验证集进行推理，提取特征向量，并保存到mia_features.csv文件中，用于训练攻击模型
- `atk1.py`：最终选用的攻击模型训练脚本，兼顾验证功能和攻击功能，是一个多层感知机用于处理mia_features中的特征向量，实现二分类
- `run_experiments.py`：不完备的批处理实验脚本，时间限制最后没有启用，仅供参考
