from prompt_few import train_prompt
import argparse
import time
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[0, 1, 2], type=list, required=False, help='随机种子序列')
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
# seed = args.seed
seed = [int(t[0]) for t in args.seed]
bsz = 64
lr = {"roberta-large": 4e-3, "roberta-base": 8e-3}
tag = f'{args.device}'
plm = ["roberta-large", "roberta-base"]

skip_list = []

path = '/home/mqh/few_save/ft/'
os.makedirs(path) if not os.path.exists(path) else None

for model in plm:
    for dz in data_sizes:
        for each in data_sizes[dz]:
            if each in skip_list:
                pass
            else:
                count = []
                for sd in seed:
                    result = train_prompt(task=each,
                             rcs=data_sizes[dz][each]['rec_step'],
                             seed=sd,
                             class_num=2,
                             bsz=bsz,
                             lr=lr[model],
                             epoch=data_sizes[dz][each]['epoch'],
                             plm=model,
                             tag=tag,
                             pre_seq_len=40,
                             data_size=dz
                             )
                    count.append(result)
                    time.sleep(1)
                avg = np.mean(count)
                std = np.std(count, ddof = 1 )
                print(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
                os.makedirs(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
