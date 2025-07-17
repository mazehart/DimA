# DimA

[English](README_EN.md) | 中文

本项目实现了DimA（维度增强，Dimension Augmentation），这是一种新颖的参数高效微调技术，用于大型预训练语言模型的高效适应。项目还包含了与其他主流PEFT方法的全面对比实验。

## 🎯 项目概述

DimA通过在注意力机制和前馈网络中引入少量可训练的增强维度来实现参数高效的模型微调。相比传统的全参数微调，DimA能够用极少的参数实现相近甚至更好的性能。

### 支持的方法
- **DimA (Augmentation)**: 本项目提出的新方法
- **Fine-tuning**: 传统全参数微调
- **LoRA**: 低秩适应方法
- **Adapter**: 适配器方法
- **Prompt Tuning**: 提示调优
- **Bias Tuning**: 偏置参数调优

### 支持的模型
- **GPT-2** 系列 (gpt2, gpt2-medium, gpt2-large)
- **RoBERTa** 系列 (roberta-base, roberta-large)

### 支持的任务
- **文本生成**: XSum摘要任务 (GPT-2)
- **文本分类**: GLUE基准任务 (RoBERTa)
  - COLA, QNLI, MRPC, RTE, QQP, MNLI-M, MNLI-MM, SST-2

## 📁 项目结构

```
DimA_full_code/
├── gpt/                    # GPT-2相关实现
│   ├── methods/           # 各种PEFT方法实现
│   │   ├── modeling_gpt2.py      # 集成DimA的GPT-2模型
│   │   ├── modeling_gpt2_lora.py # 集成LoRA的GPT-2模型
│   │   ├── adapter.py             # Adapter实现
│   │   └── aug.py                 # DimA核心模块
│   ├── train_*.py         # 各种方法的训练脚本
│   └── utils/             # 工具函数
├── roberta/                # RoBERTa相关实现
│   ├── method/            # 各种PEFT方法实现
│   ├── train_*.py         # 训练脚本
│   └── utils/             # 工具函数
├── metric/                 # 评估指标
├── *.sh                   # 训练启动脚本
└── save/                  # 模型保存目录
```

## 🚀 快速开始

### 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据下载

数据和预训练的检查点可从以下链接下载：

🔗 **数据和模型下载**: https://drive.google.com/drive/folders/1fs0qtUeyx1e61aB4Kj7kP6LgOgbZSND8?usp=sharing

下载内容包括：
- `data.zip`: 所有实验数据集的预处理版本
- `save.zip`: 预训练好的模型检查点

解压后放置在项目根目录：
```bash
# 解压数据
unzip data.zip
unzip save.zip
```

### 训练模型

#### RoBERTa模型训练

```bash
# 1. DimA方法训练
python roberta/train_aug.py --device 0 --seed 1 2 3

# 2. Fine-tuning基线
bash 1_train_ft.sh

# 3. LoRA方法
bash 2_train_lora.sh

# 4. Adapter方法
bash 3_train_adapter_few.sh

# 5. Prompt Tuning
bash 8_train_prompt_few.sh

# 6. Bias Tuning
bash 10_train_bt_few.sh
```

#### GPT-2模型训练

```bash
# DimA方法
python gpt/train_aug.py --device 0

# Fine-tuning
python gpt/train_ft.py

# LoRA
python gpt/train_lora.py --device 0 --type_m lora

# Prompt Tuning
python gpt/train_prompt.py
```

### 评估模型

```bash
# 运行测试
python test.py
```

## 🔧 配置说明

### DimA核心参数

- `aug_dim`: 增强维度大小，控制添加的参数量
- `line_m`: 线性层倍数，用于控制MLP增强
- `apply_aug_att`: 是否在注意力层应用增强
- `apply_aug_mlp`: 是否在前馈层应用增强

### 训练参数

- `lr`: 学习率，不同方法和模型大小需要调整
- `epoch`: 训练轮数
- `bsz`: 批次大小
- `seed`: 随机种子，用于结果复现

## 📊 实验结果

项目包含完整的实验对比，涵盖：

1. **参数效率对比**: DimA相比其他方法的参数使用量
2. **性能对比**: 在各个任务上的准确率比较
3. **收敛性分析**: 训练过程的损失和指标变化
4. **少样本学习**: 在有限数据下的表现

结果保存在`save/`目录下的对应子文件夹中。

## 🗂️ 文件说明

### 训练脚本命名规则

- `*_train_ft.sh`: Fine-tuning训练
- `*_train_dima.sh`: DimA方法训练  
- `*_train_lora.sh`: LoRA方法训练
- `*_train_adapter*.sh`: Adapter方法训练
- `*_train_prompt*.sh`: Prompt tuning训练
- `*_train_bt*.sh`: Bias tuning训练
- `*_few.sh`: 少样本学习实验

### 核心模块

- `aug.py`: DimA核心实现，包含AugAtt、AugMlp等模块
- `modeling_*.py`: 集成各种PEFT方法的模型实现
- `dataiter.py`: 数据迭代器
- `recorder.py`: 训练过程记录

## 📈 可视化分析

项目提供多种可视化脚本：

```bash
# 绘制训练曲线
python plot_train.py

# 绘制对比结果
python plot_new.py

# 控制向量分析
python plot_ablation_control_vector.py
```

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📞 联系

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者 52285901045@stu.ecnu.edu.cn