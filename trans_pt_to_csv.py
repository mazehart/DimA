import torch
import os
import pandas as pd

file_path = '/home/zwx21/ex_experiment/save/ft-mnli-qqp-1/'
save_path = './submission/roberta-large-new1/'
os.makedirs(save_path) if not os.path.exists(save_path) else None
dirs = [e[:-17] for e in os.listdir(file_path)]
dic_map = {
            'cola': {'name': 'CoLA.tsv', 'maps': {0: 0, 1: 1}},
           'mnli-m': {'name': 'MNLI-m.tsv', 'maps': {0: 'entailment', 1: 'neutral', 2: 'contradiction'}},
           'mnli-mm': {'name': 'MNLI-mm.tsv', 'maps': {0: 'entailment', 1: 'neutral', 2: 'contradiction'}},
           'mrpc': {'name': 'MRPC.tsv', 'maps':{0: 0, 1: 1}},
           'mrpc': {'name': 'MRPC.tsv', 'maps':{0: 'not_equivalent', 1: 'equivalent'}},
           'qnli': {'name': 'QNLI.tsv', 'maps': {0: 'entailment', 1: 'not_entailment'}},
           'qqp': {'name': 'QQP.tsv', 'maps': {0: 0, 1: 1}},
           'qqp': {'name': 'QQP.tsv', 'maps': {0: 'not_duplicate', 1: 'duplicate'}},
           'rte': {'name': 'RTE.tsv', 'maps': {0: 'entailment', 1: 'not_entailment'}},
           'sst2': {'name': 'SST-2.tsv', 'maps': {0: 'negative', 1: 'positive'}},
           'sst2': {'name': 'SST-2.tsv', 'maps': {0: 0, 1: 1}}
}

for each in dirs:

    x = torch.load(file_path + each + '_prediction_ac.pt')
    x = [dic_map[each]['maps'][y] for y in x]

    data = {'index': range(len(x)),
            'prediction': x}
    df = pd.DataFrame(data)
    df.to_csv(save_path + dic_map[each]['name'], sep="\t", index=False)
