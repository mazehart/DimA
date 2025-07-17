from prompt import train_prompt
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=True, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[1, 2], type=list, required=True, help='随机种子序列')
args = parser.parse_args()
task = {'rte': {'class_num': 2, 'epoch': 200, 'rec_step': 11},
        # 'qnli': {'class_num': 2, 'epoch': 3, 'rec_step': 150},
        # 'cola': {'class_num': 2, 'epoch': 40, 'rec_step': 25},
        # 'mrpc': {'class_num': 2, 'epoch': 60, 'rec_step': 25},
        # 'mnli-m': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
        # 'mnli-mm': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
        # 'qqp': {'class_num': 2, 'epoch': 2, 'rec_step': 150},
        # 'sst2': {'class_num': 2, 'epoch': 20, 'rec_step': 150},
        }

seed = [int(t[0]) for t in args.seed]
bsz = 64
lr = {"roberta-large": 1e-4, "roberta-base": 3e-4}
tag = f'{args.device}'
plm = ["roberta-large"]
# skip_list = ['rte', 'qnli', 'cola']
for each in task:
    for model in plm:
        for sd in seed:
            train_prompt(task=each,
                         rcs=task[each]['rec_step'],
                         seed=sd,
                         class_num=task[each]['class_num'],
                         bsz=bsz,
                         lr=lr[model],
                         epoch=task[each]['epoch'],
                         plm=model,
                         tag=tag,
                         pre_seq_len=20
                         )

            time.sleep(1)
