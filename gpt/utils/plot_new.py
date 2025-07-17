import json
import matplotlib.pyplot as plt
import os


l = [
    'E:\Augment\gpt\save\s0',
    'E:\Augment\gpt\save\s1',
    'E:\Augment\gpt\save\s2',
    'E:\Augment\gpt\save\s3',
    'E:\Augment\gpt\save\s4',
    'E:\Augment\gpt\save\s5',
]
c = ['r', 'r', 'g', 'g', 'b', 'b', 'm', 'm', 'gray', 'gray', 'purple', 'purple']
c2 = ['r', 'g', 'b', 'm', 'gray', 'purple']
plt.title('Analysis')
lines = ['-', '--'] * 10

accs = []
ents = []
acc_list = []
ec_list = []
lrs = []
for element in l:
    dic = os.listdir(element)
    acc = [element+'/'+x for x in dic if 'acc' in x and ('train' in x or 'score' in x)]
    ent = [element+'/'+x for x in dic if 'acc' not in x and ('train' in x or 'score' in x)]
    accs += acc
    ents += ent
    acc_l = [element+'/'+x for x in dic if 'acc_list' in x]
    ent_l = [element+'/'+x for x in dic if 'ec_list' in x]
    acc_list += acc_l * 2
    ec_list += ent_l* 2
    lr = [element+'/'+x for x in dic if 'lr' in x]
    lrs += lr

for each, color , line, l in zip(accs, c, lines, acc_list):

    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        x = range(y.__len__())

        plt.plot(x, y, color=color, linestyle=line)
    if 'score' in each:
        with open(l, 'r', encoding='utf-8') as f:
            acc_l = json.load(f)
            x = [e for idx, e in enumerate(x) if idx+1 in acc_l]
            y =  [e for idx, e in enumerate(y) if idx+1 in acc_l]

            plt.plot(x, y, 'o', color=color)

plt.figure()
for each, color , line, l in zip(ents, c, lines, ec_list):

    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        x = range(y.__len__())

        plt.plot(x, y, color=color, linestyle=line)

    if 'score' in each:
        with open(l, 'r', encoding='utf-8') as f:
            acc_l = json.load(f)
            x = [e for idx, e in enumerate(x) if idx+1 in acc_l]
            y =  [e for idx, e in enumerate(y) if idx+1 in acc_l]

            plt.plot(x, y, 'o', color=color)

plt.figure()
for each, color , line in zip(lrs, c2, lines):

    with open(each, 'r', encoding='utf-8') as f:
        y = json.load(f)
        y = [e[0] for e in y if isinstance(e, list)] if isinstance(y[0], list) else y
        x = range(y.__len__())

        plt.plot(x, y, color=color)
plt.legend()

plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

