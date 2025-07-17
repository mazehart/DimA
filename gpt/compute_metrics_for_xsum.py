import evaluate
import torch

temp = [
    'rouge',
    'bleu',
    'meteor',
    'ter', ]

task =['/data1/zwx/gpt2_pt.pt',
       '/data1/zwx/gpt2-large_pt.pt',
       ]
name = ['gpt2_pt', 'gpt2-large_pt']
for each in temp:
    metric = evaluate.load(each)
    for x, y in zip(task, name):
        test_text = torch.load(x)
        result = torch.load('/data1/zwx/ex_experiment/data/xsum/test_label.pt')
        print(y)
        print(metric.compute(predictions=test_text, references=result))