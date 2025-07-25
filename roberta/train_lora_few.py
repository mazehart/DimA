from lora_few import train_lora
import argparse
import time
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=True, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[0, 1, 2], type=list, required=True, help='随机种子序列')
parser.add_argument('--typem', default='lora', type=str, required=True, help='adapter or lora')
args = parser.parse_args()
data_sizes = {
    50: {'mrpc': {'epoch': 2000, 'rec_step': 1},
         'cola': {'epoch': 2000, 'rec_step': 1},
         'rte': {'epoch': 2000, 'rec_step': 1},
         'sst2': {'epoch': 2000, 'rec_step': 1},
         },
    100: {'mrpc': {'epoch': 1000, 'rec_step': 3},
          'cola': {'epoch': 1000, 'rec_step': 3},
          'rte': {'epoch': 1000, 'rec_step': 3},
          'sst2': {'epoch': 1000, 'rec_step': 3},
          },
    200: {'mrpc': {'epoch': 500, 'rec_step': 3},
          'cola': {'epoch': 500, 'rec_step': 3},
          'rte': {'epoch': 500, 'rec_step': 3},
          'sst2': {'epoch': 500, 'rec_step': 3},
          },
    400: {'mrpc': {'epoch': 250, 'rec_step': 3},
          'cola': {'epoch': 250, 'rec_step': 3},
          'rte': {'epoch': 250, 'rec_step': 3},
          'sst2': {'epoch': 250, 'rec_step': 3},
          }}
seed = [int(t[0]) for t in args.seed]
typem = args.typem
bsz = 64
plm = ["roberta-large"]
lr = {"roberta-large": 3e-04, "roberta-base": 6e-04} if typem == 'lora' else {
    "roberta-large": 3e-04, "roberta-base": 6e-04}
tag = f'{args.device}'

skip_list = []

path = f'/data1/zwx/few_save/{typem}/'
os.makedirs(path) if not os.path.exists(path) else None

for model in plm:
    for dz in data_sizes:
        for each in data_sizes[dz]:
            if each in skip_list:
                pass
            else:
                count = []
                for sd in seed:
                    result = train_lora(task=each,
                           rec_step=data_sizes[dz][each]['rec_step'],
                           seed=sd,
                           class_num=2,
                           bsz=bsz,
                           lr=lr[model],
                           epoch=data_sizes[dz][each]['epoch'],
                           plm=model,
                           tag=tag,
                           type_m=typem,
                           data_size=dz)
                    count.append(result)
                    time.sleep(1)
                avg = np.mean(count)
                std = np.std(count, ddof = 1 )
                print(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
                os.makedirs(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
                