from lora import train_lora
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='0', type=str, required=True, help='设置使用哪些显卡')
parser.add_argument('--seed', nargs='+', default=[1, 2], type=list, required=True, help='随机种子序列')
parser.add_argument('--typem', default='lora', type=str, required=True, help='adapter or lora')
args = parser.parse_args()
task = {
    'mnli-m': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
    'mnli-mm': {'class_num': 3, 'epoch': 2, 'rec_step': 150},
    'rte': {'class_num': 2, 'epoch': 40, 'rec_step': 11},
    'qqp': {'class_num': 2, 'epoch': 2, 'rec_step': 150},
    'qnli': {'class_num': 2, 'epoch': 3, 'rec_step': 150},
    'cola': {'class_num': 2, 'epoch': 40, 'rec_step': 25},
    'mrpc': {'class_num': 2, 'epoch': 60, 'rec_step': 25},
    'sst2': {'class_num': 2, 'epoch': 20, 'rec_step': 150},
}

seed = [int(t[0]) for t in args.seed]
typem = args.typem
bsz = 64
plm = ["roberta-large", "roberta-base"]
lr = {"roberta-large": 3e-04, "roberta-base": 6e-04} if typem == 'lora' else {
    "roberta-large": 3e-04, "roberta-base": 6e-04}
tag = f'{args.device}'
# run_list = {0: ['mnli-m', 'mnli-mm', 'rte', 'qqp', 'qnli', 'cola', 'mrpc', 'sst2'],
#             1: ['mnli-mm', 'mrpc', 'sst2'], 2: ['mnli-mm', 'qnli', 'cola', 'mrpc', 'sst2']}
for each in task:
    for model in plm:
        for sd in seed:
            # if each in run_list[sd]:
            train_lora(task=each,
                       rec_step=task[each]['rec_step'],
                       seed=sd,
                       class_num=task[each]['class_num'],
                       bsz=bsz,
                       lr=lr[model],
                       epoch=task[each]['epoch'],
                       plm=model,
                       tag=tag,
                       type_m=typem)
            # else:
            #     pass

            time.sleep(1)
