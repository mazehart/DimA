# DimA

"The code will be updated shortly."

# Appendix A. Hyperparameter Settings

This chapter presents the hyperparameter settings for different tasks and models in the experiments. The tables below detail the parameters for different task settings in the scenarios of single-task fine-tuning and Few-shot learning, as well as the learning rate settings for different methods.

## Single Task

| Task     | RTE | MRPC | CoLA | SST-2 | QNLI | MNLI-m | MNLI-mm | QQP | XSUM |
|----------|-----|------|------|-------|------|--------|---------|-----|------|
| max length | 128 | 128  | 128  | 128   | 128  | 128    | 128     | 128 | 256  |
| batch size | 64  | 64   | 64   | 64    | 64   | 64     | 64      | 64  | 4    |
| epochs    | 40  | 60   | 40   | 20    | 3    | 2      | 2       | 2   | 1    |
| save steps| 11  | 25   | 25   | 150   | 150  | 150    | 150     | 150 | 150  |

*The settings of the dataset under single-task fine-tuning are as follows: "max length" refers to the length at which text is truncated, chosen based on the typical length of task text and learning efficiency; "save steps" indicates the number of steps at which the model's performance is saved based on the validation set.*

## Few-shot

| Task | Data Size |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|      | 50        | 100  | 200  | 400  | 50   | 100  | 200  | 400  | 50   | 100  | 200  | 400  | 50   | 100  | 200  | 400  |
|------|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| max length | 128 | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  | 128  |
| batch size | 64  | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   | 64   |
| epochs    | 2000| 1000 | 500  | 250  | 2000 | 1000 | 500  | 250  | 2000 | 1000 | 500  | 250  | 2000 | 1000 | 500  | 250  |
| save steps| 1   | 3    | 3    | 3    | 1    | 3    | 3    | 3    | 1    | 3    | 3    | 3    | 1    | 3    | 3    | 3    |

*The hyperparameter settings of the dataset under Few-shot are as follows: "data size" refers to the different partition sizes of the dataset.*

## Learning Rate

| Models | Size | RoBERTa     |         |         | GPT2    |        |        |
|--------|------|-------------|---------|---------|---------|--------|--------|
|        |      | base        | large   | base    | medium  | large  |
|--------|------|-------------|---------|---------|---------|--------|--------|
| FT     |      | 4e-05       | 3e-05   | 2e-05   | 1e-05   | 2e-05  | 3e-05  |
| BitFit |      | 2e-04       | 1e-04   | 1e-04   | 1e-04   | 2e-04  | 3e-05  |
| Adapter|      | 8e-04       | 4e-04   | 2e-04  |        | 1e-03 | 5e-04  | 2e-04  |
| LoRA   |      | 8e-04       | 4e-04   | 4e-04   | 1e-03   | 5e-04  | 2e-04  |
| P-Tuning|     | 3e-04       | 1.2e-04 | 1.2e-04 | 3e-03   | 4e-04  | 2e-04  |
| AdapterFusion| | -          | -       | 4e-04   | -       | -      | -      |
| DimA   |      | 8e-04       | 4e-04   | 4e-04   | 2e-03   | 3e-04  | 1e-04  |

*The learning rate settings for different methods. "AdapterFusion" utilizes the module weights learned by the "Adapter" during single-task training and only conducts knowledge transfer experiments in the Few-shot setting.*

