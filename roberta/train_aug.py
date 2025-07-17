from augment import train_aug
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=True, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[1, 2, 3], type=list, required=True, help='随机种子序列')
args = parser.parse_args()
task = {
    'mrpc': {'class_num': 2, 'epoch': 60, 'rec_step': 25},
    'mnli-m': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
    'qnli': {'class_num': 2, 'epoch': 3, 'rec_step': 150},
    'cola': {'class_num': 2, 'epoch': 40, 'rec_step': 25},
    'rte': {'class_num': 2, 'epoch': 40, 'rec_step': 11},
    'mnli-mm': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
    'qqp': {'class_num': 2, 'epoch': 2, 'rec_step': 150},
    'sst2': {'class_num': 2, 'epoch': 20, 'rec_step': 150},
}

seed = [int(t[0]) for t in args.seed]
bsz = 64
plm = [
    "roberta-large",
    "roberta-base"
]
add_dim = {
    9: {"roberta-large": 0.00015, "roberta-base": 0.0003},
    3: {"roberta-large": 0.00025, "roberta-base": 0.0005},
    1: {"roberta-large": 0.0004, "roberta-base": 0.0008}
}
tag = f'{args.device}'
line_m = 1
train_list = ['qqp', 'sst2']
for each in task:
    for model in plm:
        for dim in add_dim:
            for sd in seed:
                if each in train_list:
                    train_aug(task=each,
                              rec_step=task[each]['rec_step'],
                              seed=sd,
                              class_num=task[each]['class_num'],
                              bsz=bsz,
                              lr=add_dim[dim][model],
                              add_dim=dim,
                              epoch=task[each]['epoch'],
                              plm=model,
                              tag=tag,
                              line_m=line_m,
                              )
                else:
                    pass

                time.sleep(1)
