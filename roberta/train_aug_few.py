from augment_few import train_aug
import argparse
import time
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[0, 1, 2], type=list, required=False, help='随机种子序列')
args = parser.parse_args()

seed = [int(t[0]) for t in args.seed]
bsz = 64
plm = ["roberta-large", "roberta-base"]
lr = {"roberta-large": 0.0003, "roberta-base": 0.0006}
tag = f'{args.device}'
line_m = 1
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
# skip_list = ['mrpc', 'cola']
# skip_list = ['rte', 'sst2']
skip_list = []

path = '/data1/zwx/few_save/aug/'
os.makedirs(path) if not os.path.exists(path) else None

for model in plm:
    for dz in data_sizes:
        for each in data_sizes[dz]:
            if each in skip_list or dz in skip_list:
                pass
            else:
                count = []
                for sd in seed:
                    result = train_aug(task=each,
                          rec_step=data_sizes[dz][each]['rec_step'],
                          seed=sd,
                          class_num=2,
                          bsz=bsz,
                          lr=lr[model],
                          add_dim=1,
                          epoch=data_sizes[dz][each]['epoch'],
                          plm=model,
                          tag=tag,
                          line_m=line_m,
                          data_size=dz,
                          )
                    count.append(result)
                    time.sleep(1)
                avg = np.mean(count)
                std = np.std(count, ddof = 1 )
                print(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
                os.makedirs(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')