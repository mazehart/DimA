from adapter import train_ad
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--type_m', default='lora', type=str, required=False, help='lora or ada')
args = parser.parse_args()
task = {'xsum': {'epoch': 1, 'rec_step': 150},
        }
seed = 0
bsz = 16
plm = [
    "gpt2", 
    # "gpt2-medium",
    #    "gpt2-large"
       ]
lr = {"gpt2": 0.001, "gpt2-medium": 0.0005, "gpt2-large":0.0002}
tag = f'{args.device}'

for each in task:
    for model in plm:
        train_ad(task=each,
                 rec_step=task[each]['rec_step'],
                 seed=seed,
                 bsz=bsz,
                 lr=lr[model],
                 type_m=args.type_m,
                 epoch=task[each]['epoch'],
                 plm=model,
                 tag=tag,
                 )
        time.sleep(1)
