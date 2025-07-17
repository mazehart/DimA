import json
import matplotlib.pyplot as plt
import os

l = [
    '/home/mqh/ex_experiment/save/few_shot/mrpc/seed_0/roberta-large/Bt_lr_0.0001_line_2000_bsz_64_step_1_0.0007687784406195268_time_225.56885719299316',
    '/home/mqh/ex_experiment/save/few_shot/mrpc/seed_0/roberta-large/Ft_lr_3e-06_line_2000_bsz_64_step_1_time_387.41403245925903',
    '/home/mqh/ex_experiment/save/few_shot/mrpc/seed_0/roberta-large/Augment_line_2000_bsz_64_step_1_0.006619075166912376',
]
c = ['r', 'r', 'g', 'g', 'b', 'b', 'm', 'm', 'gray', 'gray', 'purple', 'purple', 'yellow', 'yellow', 'black', 'black']
c2 = ['r', 'g', 'b', 'm', 'gray', 'purple']
plt.title('Anlysis')
lines = ['-', '--'] * 10

accs = []
ents = []
acc_list = []
ec_list = []
lrs = []
for element in l:
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

for each, color, line, l in zip(accs, c, lines, acc_list):

    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        x = range(y.__len__())

        plt.plot(x, y, color=color, linestyle=line)
    if 'score' in each:
        with open(l, 'r', encoding='utf-8') as f:
            acc_l = json.load(f)
            x = [e for idx, e in enumerate(x) if idx + 1 in acc_l]
            y = [e for idx, e in enumerate(y) if idx + 1 in acc_l]

            plt.plot(x, y, 'o', color=color)
        plt.savefig('/home/mqh/ex_experiment/save/myplot_0.png')

plt.figure()
for each, color, line, l in zip(ents, c, lines, ec_list):

    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        x = range(y.__len__())

        plt.plot(x, y, color=color, linestyle=line)

    if 'score' in each:
        with open(l, 'r', encoding='utf-8') as f:
            acc_l = json.load(f)
            x = [e for idx, e in enumerate(x) if idx + 1 in acc_l]
            y = [e for idx, e in enumerate(y) if idx + 1 in acc_l]

            plt.plot(x, y, 'o', color=color)
        plt.savefig('/home/mqh/ex_experiment/save/myplot_1.png')

plt.figure()
for each, color, line in zip(lrs, c2, lines):
    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        y = [e[0] for e in y if isinstance(e, list)] if isinstance(y[0], list) else y
        x = range(y.__len__())

        plt.plot(x, y, color=color)
plt.legend()

plt.xlabel('iteration times')
plt.ylabel('rate')
plt.savefig('/home/mqh/ex_experiment/save/myplot.png')
