# DimA

English | [中文](README.md)

This project implements DimA (Dimension Augmentation), a novel parameter-efficient fine-tuning technique for large pre-trained language models. The project also includes comprehensive comparative experiments with other mainstream PEFT methods.

## 🎯 Project Overview

DimA achieves parameter-efficient model fine-tuning by introducing a small number of trainable augmentation dimensions in attention mechanisms and feed-forward networks. Compared to traditional full-parameter fine-tuning, DimA can achieve comparable or even better performance with extremely few parameters.

### Supported Methods
- **DimA (Augmentation)**: The novel method proposed in this project
- **Fine-tuning**: Traditional full-parameter fine-tuning
- **LoRA**: Low-Rank Adaptation
- **Adapter**: Adapter methods
- **Prompt Tuning**: Prompt-based tuning
- **Bias Tuning**: Bias parameter tuning

### Supported Models
- **GPT-2** series (gpt2, gpt2-medium, gpt2-large)
- **RoBERTa** series (roberta-base, roberta-large)

### Supported Tasks
- **Text Generation**: XSum summarization task (GPT-2)
- **Text Classification**: GLUE benchmark tasks (RoBERTa)
  - COLA, QNLI, MRPC, RTE, QQP, MNLI-M, MNLI-MM, SST-2

## 📁 Project Structure

```
DimA_full_code/
├── gpt/                    # GPT-2 related implementations
│   ├── methods/           # Various PEFT method implementations
│   │   ├── modeling_gpt2.py      # GPT-2 model with DimA integration
│   │   ├── modeling_gpt2_lora.py # GPT-2 model with LoRA integration
│   │   ├── adapter.py             # Adapter implementation
│   │   └── aug.py                 # DimA core modules
│   ├── train_*.py         # Training scripts for various methods
│   └── utils/             # Utility functions
├── roberta/                # RoBERTa related implementations
│   ├── method/            # Various PEFT method implementations
│   ├── train_*.py         # Training scripts
│   └── utils/             # Utility functions
├── metric/                 # Evaluation metrics
├── *.sh                   # Training launch scripts
└── save/                  # Model checkpoint directory
```

## 🚀 Quick Start

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Data Download

Data and pre-trained checkpoints can be downloaded from the following link:

🔗 **Data and Model Download**: https://drive.google.com/drive/folders/1fs0qtUeyx1e61aB4Kj7kP6LgOgbZSND8?usp=sharing

Download contents include:
- `data.zip`: Preprocessed versions of all experimental datasets
- `save.zip`: Pre-trained model checkpoints

Extract and place in the project root directory:
```bash
# Extract data
unzip data.zip
unzip save.zip
```

### Model Training

#### RoBERTa Model Training

```bash
# 1. DimA method training
python roberta/train_aug.py --device 0 --seed 1 2 3

# 2. Fine-tuning baseline
bash 1_train_ft.sh

# 3. LoRA method
bash 2_train_lora.sh

# 4. Adapter method
bash 3_train_adapter_few.sh

# 5. Prompt Tuning
bash 8_train_prompt_few.sh

# 6. Bias Tuning
bash 10_train_bt_few.sh
```

#### GPT-2 Model Training

```bash
# DimA method
python gpt/train_aug.py --device 0

# Fine-tuning
python gpt/train_ft.py

# LoRA
python gpt/train_lora.py --device 0 --type_m lora

# Prompt Tuning
python gpt/train_prompt.py
```

### Model Evaluation

```bash
# Run tests
python test.py
```

## 🔧 Configuration

### DimA Core Parameters

- `aug_dim`: Augmentation dimension size, controls the number of added parameters
- `line_m`: Linear layer multiplier, used to control MLP augmentation
- `apply_aug_att`: Whether to apply augmentation in attention layers
- `apply_aug_mlp`: Whether to apply augmentation in feed-forward layers

### Training Parameters

- `lr`: Learning rate, needs adjustment for different methods and model sizes
- `epoch`: Number of training epochs
- `bsz`: Batch size
- `seed`: Random seed for reproducibility

## 📊 Experimental Results

The project includes comprehensive experimental comparisons covering:

1. **Parameter Efficiency Comparison**: Parameter usage of DimA vs. other methods
2. **Performance Comparison**: Accuracy comparison across various tasks
3. **Convergence Analysis**: Loss and metric changes during training
4. **Few-shot Learning**: Performance under limited data conditions

Results are saved in corresponding subdirectories under the `save/` directory.

## 🗂️ File Description

### Training Script Naming Convention

- `*_train_ft.sh`: Fine-tuning training
- `*_train_dima.sh`: DimA method training  
- `*_train_lora.sh`: LoRA method training
- `*_train_adapter*.sh`: Adapter method training
- `*_train_prompt*.sh`: Prompt tuning training
- `*_train_bt*.sh`: Bias tuning training
- `*_few.sh`: Few-shot learning experiments

### Core Modules

- `aug.py`: DimA core implementation, including AugAtt, AugMlp modules
- `modeling_*.py`: Model implementations integrating various PEFT methods
- `dataiter.py`: Data iterator
- `recorder.py`: Training process recording

## 📈 Visualization Analysis

The project provides various visualization scripts:

```bash
# Plot training curves
python plot_train.py

# Plot comparison results
python plot_new.py

# Control vector analysis
python plot_ablation_control_vector.py
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📞 Contact

For questions or suggestions, please contact via:
- Submit GitHub Issues
- Email project maintainers 52285901045@stu.ecnu.edu.cn
