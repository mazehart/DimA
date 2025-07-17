import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_train(ax, dics):

    c = ['r', 'r', 'g', 'g', 'b', 'b']
    c2 = ['r', 'g', 'b']

    accs = []
    ents = []
    acc_list = []
    ec_list = []
    lrs = []
    for element in dics:
        dic = os.listdir(element)
        acc = [element + '/' + x for x in dic if 'acc' in x and ('train' in x or 'score' in x)]
        ent = [element + '/' + x for x in dic if 'acc' not in x and ('train' in x or 'score' in x)]
        accs += acc
        ents += ent
        acc_l = [element + '/' + x for x in dic if 'acc_list' in x]
        ent_l = [element + '/' + x for x in dic if 'ec_list' in x]
        acc_list += acc_l * 2
        ec_list += ent_l * 2
        lr = [element + '/' + x for x in dic if 'lr' in x]
        lrs += lr
    ax.grid(True, linewidth=0.1, color='gray')
    for each, color, l in zip(accs, c, acc_list):

        with open(each, 'r', encoding='utf-8') as f:
            y = json.load(f)
            x = range(y.__len__())

            ax.plot(x, y, color=color, linestyle='-' if 'score' in each else '--')
        if 'score' in each:
            with open(l, 'r', encoding='utf-8') as f:
                acc_l = json.load(f)
                x = [e for idx, e in enumerate(x) if idx + 1 in acc_l]
                y = [e for idx, e in enumerate(y) if idx + 1 in acc_l]

                ax.plot(x, y, 'o', color=color)
            

sizes = ['base', 'large']
tasks = ['rte', 'mrpc',  'cola','sst2']

for size in sizes:
    fig, axs = plt.subplots(4, 4, figsize=(30, 20))
    for idx1, taskk in enumerate(tasks):
        for idx2, ds in enumerate([2000, 1000, 500, 250]):
            task = [f'/data1/zwx/save/mul_zero/{taskk}/seed_{seed}/roberta-{size}/Augment_line_{ds}' for seed in [0, 1, 2]]
            task_aug = [f'/data1/zwx/ex_experiment/save/few_shot/{taskk}/seed_{seed}/roberta-{size}/Augment_line_{ds}' for seed in [0, 1, 2]]

            # 创建一个4x4的子图网格
            ax = axs[idx2][idx1]
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{taskk} k={100000//ds}')
            # 创建上下两张比例为9:3的小图，设置坐标轴范围
            ax_top = ax.inset_axes([0, 0.53, 1, 0.47], ylim=(0, 1.1))
            ax_bottom = ax.inset_axes([0, 0, 1, 0.47], ylim=(0, 1.1))
            plot_train(ax=ax_top, dics=task)
            plot_train(ax=ax_bottom, dics=task_aug)
    plt.tight_layout()
    plt.savefig(f'/data1/zwx/ex_experiment//{size}_train.pdf')
    plt.close()
