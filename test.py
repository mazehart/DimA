import torch
from utils.dataiter import DataIter
from method.modeling_roberta_mul_app import RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaConfig
import json
from evaluate import load as load_metric

def test(task, ds, size, merge_list, wt, sd, metric=None, tag=0,):
    plm=f"roberta-{size}"
    class_num = 2
    bsz = 128
    config = RobertaConfig.from_pretrained(plm)
    config.merge = False
    config.merge_list = merge_list
    config.merge_dim = config.merge_list.__len__() * config.num_attention_heads
    config.num_labels = class_num
    config.attention_dim = config.hidden_size // config.num_attention_heads
    device = f'cuda:{tag}' if torch.cuda.is_available() else 'cpu'
    config.device = device
    config.wt = wt
    tok = RobertaTokenizer.from_pretrained(plm)
    model = RobertaForSequenceClassification.from_pretrained(plm, config=config)
    model = model.merge()
    model.load(f'/home/mqh/save/mul_zero/{task}/seed_{sd}/{plm}/Augment_line_{ds}/Augment_{task}best_acc_')
    
    dev_data = torch.load(f'./data/{task}/validation_data.pt')
    dev_label = torch.load(f'./data/{task}/validation_label.pt')
    dev_data = DataIter(tok, dev_data, dev_label, batch_size=bsz, max_length=128)
         
    model.eval()
    pre = []
    ref = []
    with torch.no_grad():
        for dev_step, dev_batch in enumerate(dev_data):
            output = model(input_ids=dev_batch[0]['input_ids'].to(device),
                        labels=dev_batch[1].to(device),
                        attention_mask=dev_batch[0]['attention_mask'].to(device))
            
            # p = output.logits.argmax(dim=-1)
            # pre += p.view(-1).tolist()
            # ref += dev_batch[1].view(-1).tolist()
            p = [each[idx] for each, idx in zip(torch.softmax(output.logits, dim=-1), dev_batch[1])]
            pre += p
        p = torch.mean(torch.tensor(pre))
        # p = metric.compute(predictions=pre, references=ref)['accuracy']
    return p


if __name__ == '__main__':
    size = 'base'
    task = 'cola'
    l = ['mnli-m', 'mnli-mm', 'qnli', 'qqp', task]
    # metric = load_metric('accuracy')
    for ds in [400]:
        for each in range(5):
            pre = []
            for weight in range(40, 0, -1):
                p = 0
                weight = weight / 20
                wt = torch.ones(5)
                wt[each] = weight
                for sd in [0, 1, 2]:
                    merge_list = [f'/home/mqh/ex_experiment/save/multi/{size}/Augment_mnli-mbest_acc_param.pt',
                        f'/home/mqh/ex_experiment/save/multi/{size}/Augment_mnli-mmbest_acc_param.pt',
                        f'/home/mqh/ex_experiment/save/multi/{size}/Augment_qnlibest_acc_param.pt',
                        f'/home/mqh/ex_experiment/save/multi/{size}/Augment_qqpbest_acc_param.pt',
                        f'/home/mqh/save/mul_zero/{task}/seed_{sd}/roberta-{size}/Augment_line_{ds}/Augment_{task}best_acc_param.pt']
                    
                    p += test(task=task, merge_list=merge_list, wt=wt, size=size, ds=ds, sd=sd, tag=1, 
                            #   metric=metric
                            )
                pre.append(float(p/3))
                print(pre)
            
            with open(f"/home/mqh/save/mul_zero/{size}_{task}_{ds}_{l[each]}.json", "w") as outfile:
                json.dump(pre, outfile)
