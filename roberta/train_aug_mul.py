from augment_mul import train_aug
import argparse
import time
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[1, 2, 3], type=list, required=False, help='随机种子序列')
args = parser.parse_args()
merge_list = {
    # "roberta-large": [
    # '/data1/zwx/ex_experiment/save/multi/large/Augment_mnli-mbest_acc_param.pt',
    # '/data1/zwx/ex_experiment/save/multi/large/Augment_mnli-mmbest_acc_param.pt',
    # '/data1/zwx/ex_experiment/save/multi/large/Augment_qnlibest_acc_param.pt',
    # '/data1/zwx/ex_experiment/save/multi/large/Augment_qqpbest_acc_param.pt'
    # ],
    "roberta-base": [
        '/data1/zwx/ex_experiment/save/multi/base/Augment_mnli-mbest_acc_param.pt',
        '/data1/zwx/ex_experiment/save/multi/base/Augment_mnli-mmbest_acc_param.pt',
        '/data1/zwx/ex_experiment/save/multi/base/Augment_qnlibest_acc_param.pt',
        '/data1/zwx/ex_experiment/save/multi/base/Augment_qqpbest_acc_param.pt'
        ]}

seed = [int(t[0]) for t in args.seed]
# seed = args.seed
bsz = 64
# plm = {
#     # "roberta-large": 64, 
#        "roberta-base": 48
#        }
plm = {
    # "roberta-large": 48,  
    "roberta-base": 36}
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
          }
}
skip_list = ['mrpc', 'rte', 'sst2', 'wnli']
# skip_list = []
# skip_list = []
path = '/data1/zwx/save/mul_zero_new/'
os.makedirs(path) if not os.path.exists(path) else None

for model in plm:
    for dz in data_sizes:
        for each in data_sizes[dz]:
            if each in skip_list:
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
                            merge_dim=plm[model],
                            merge_list=merge_list[model],
                            data_size=dz,
                            merge_lr=0.2 if model == 'roberta-base' else 0.1)
                    count.append(result)
                    time.sleep(1)
                avg = np.mean(count)
                std = np.std(count, ddof = 1 )
                print(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
                os.makedirs(path + f'{model}/task_{each}_dz_{dz}_std_{std}_mean_{avg}')
