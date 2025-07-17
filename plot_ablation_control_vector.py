import json
import matplotlib.pyplot as plt
import numpy as np

sizes = ['base', 'large']
tasks = ['rte', 'mrpc',  'cola','sst2']

for size in sizes:
    with open(f'/home/mqh/mul_zero/{size}_dict.json', 'r', encoding='utf-8') as file:
         bar = json.load(file)
    fig, axs = plt.subplots(4, 4, figsize=(30, 20))
    for idx1, taskk in enumerate(tasks):
        ll = ['mnli-m', 'mnli-mm', 'qnli', 'qqp', taskk]
        for idx2, ds in enumerate([2000, 1000, 500,250 ]):
            task = f'{size}_{taskk}_{ds}'
            l = [
                f'/home/mqh/mul_zero/{task}_mnli-m.json',
                f'/home/mqh/mul_zero/{task}_mnli-mm.json',
                f'/home/mqh/mul_zero/{task}_qnli.json',
                f'/home/mqh/mul_zero/{task}_qqp.json',
                f'/home/mqh/mul_zero/{task}_{taskk}.json'
            ]
            c2 = ['r', 'g', 'b', 'm', 'gray']
            bar_temp = bar[task]
            # 创建一个4x4的子图网格
            ax = axs[idx2][idx1]
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{taskk} k={100000//ds}')
            # 创建上下两张比例为9:3的小图，设置坐标轴范围
            ax_top = ax.inset_axes([0, 0.53, 1, 0.47])
            ax_top.set_title
            ax_bottom = ax.inset_axes([0, 0, 1, 0.47], ylim=(0, 0.9))
            ax_bottom.bar(x = range(5), yerr=bar_temp[1], color=c2, align='center', alpha=0.5,height=bar_temp[0])
            ax_bottom.set_xticks(range(5))
            ax_bottom.set_xticklabels(ll)
            for each, color in zip(l, c2):
                with open(each, 'r', encoding='utf-8') as f:
                    y = json.load(f)
                    x = np.arange(0, 40, 1)
                    ax_top.plot(x, y, color=color, label=each.split('_')[4].split('.')[0])
                    ax_top.set_xticks(np.arange(0, 40.00001, 10))
                    ax_top.set_xticklabels(['x2.0','x1.5', 'x1.0', 'x0.5','x0.0'])
            ax_top.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(f'/home/mqh/ex_experiment/{size}.pdf')
    plt.close()
