from augment import train_aug
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=True, help='设置使用哪些显卡')
args = parser.parse_args()
task = {'xsum': {'epoch': 1, 'rec_step': 150},
        }

seed = 0
bsz = 16
plm = ["gpt2"]
add_dim = {
           1: {"gpt2": 0.002, "gpt2-medium": 0.003, "gpt2-large": 0.0009}}

tag = f'{args.device}'
line_m = 1
for each in task:
    for model in plm:
        for dim in add_dim:
            train_aug(task=each,
                        rec_step=task[each]['rec_step'],
                        seed=seed,
                        bsz=bsz,
                        lr=add_dim[dim][model],
                        add_dim=dim,
                        epoch=task[each]['epoch'],
                        plm=model,
                        tag=tag,
                        )
           