from method.roberta import RobertaForSequenceClassification
import torch
from transformers import RobertaConfig, RobertaTokenizer, get_linear_schedule_with_warmup
from utils.dataiter import DataIter
from torch.optim import AdamW
from utils.recorder import Recorder
from evaluate import load as load_metric
import os
from method.count_param import count_param


def train_lora(task, seed, class_num, bsz, lr, epoch, plm, tag, rec_step, type_m, data_size):
    config = RobertaConfig.from_pretrained(plm)
    config.type_m = type_m
    if type_m == 'lora':
        config.apply_lora = True
        config.apply_adapter = False
    else:
        config.apply_lora = False
        config.apply_adapter = True
    config.lora_r = 24
    config.lora_alpha = 16
    # config.adapter_type = 'pfeiffer'
    config.adapter_type = 'houlsby'
    config.adapter_size = 32
    config.epoch = epoch
    config.lr = lr
    config.num_labels = class_num
    
    
    torch.manual_seed(seed)
    device = f'cuda:{tag}' if torch.cuda.is_available() else 'cpu'
    tok = RobertaTokenizer.from_pretrained(plm)
    model = RobertaForSequenceClassification.from_pretrained(plm, config=config).to(device)
    metric = load_metric('accuracy')
    train_data = torch.load(f'./data/{task}/train_{data_size}_data.pt')
    train_label = torch.load(f'./data/{task}/train_{data_size}_label.pt')
    dev_data = torch.load(f'./data/{task}/validation_{data_size}_data.pt')
    dev_label = torch.load(f'./data/{task}/validation_{data_size}_label.pt')
    train_data = DataIter(tok, train_data, train_label, batch_size=bsz, max_length=128)
    dev_data = DataIter(tok, dev_data, dev_label, batch_size=bsz, max_length=128)
    
    train_param = [value for param, value in model.named_parameters() if type_m in param or 'class' in param]
    optim = AdamW(params=train_param, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=epoch * (train_data.batch_num+1) // 3,
                                                num_training_steps=epoch * (train_data.batch_num+1))
    model.train()
    param = count_param(model=model, param_name=type_m)
    file_path = f'./save/few_shot/{task}/seed_{seed}/{plm}/{type_m}_lr_{lr}_line_{epoch}_bsz_{bsz}_step_{rec_step}_{param}'
    os.makedirs(file_path) if not os.path.exists(file_path) else print('path exits!!!')
    rec = Recorder(save_path=file_path + f'/{type_m}_{task}', record_steps=rec_step)
    rec.start()

    for i in range(config.epoch):

        rec.epoch_start(i)

        for train_step, batch in enumerate(train_data):

            output = model(input_ids=batch[0]['input_ids'].to(device),
                           labels=batch[1].to(device),
                           attention_mask=batch[0]['attention_mask'].to(device))
            loss = output.loss
            p = output.logits.argmax(dim=-1)
            loss.backward()
            optim.step()
            scheduler.step()

            model.zero_grad()
            rec.record_train(train_step, loss,
                             metric.compute(predictions=p, references=batch[1].to(device))['accuracy'])

            if train_step % rec.record_steps == 0 and train_step != 0:
                with torch.no_grad():
                    model.eval()
                    sum_loss = 0
                    pre = []
                    ref = []
                    for dev_step, dev_batch in enumerate(dev_data):
                        output = model(input_ids=dev_batch[0]['input_ids'].to(device),
                                       labels=dev_batch[1].to(device),
                                       attention_mask=dev_batch[0]['attention_mask'].to(device))
                        loss = output.loss
                        p = output.logits.argmax(dim=-1)
                        sum_loss += loss
                        pre += p.view(-1).tolist()
                        ref += dev_batch[1].view(-1).tolist()
                    if dev_data.batch_num > 0:
                        sum_loss /= dev_data.batch_num
                    sum_p = metric.compute(predictions=pre, references=ref)['accuracy']
                    save_dir = rec.record_dev(train_step, sum_loss, sum_p)
                    for dirt in save_dir:
                        model.save(dirt) if dirt is not None else None
                    rec.save_report()
                    print(rec.report)
                    print(scheduler.get_last_lr())
                    rec.rec_lr(scheduler.get_last_lr())
                    
            model.train()
        if rec.early_stop():
            break
    time = rec.time()
    os.rename(file_path, file_path + f'_time_{time}')
    file_path = file_path + f'_time_{time}'
    test_data = torch.load(f'./data/{task}/validation_data.pt')
    test_label = torch.load(f'./data/{task}/validation_label.pt')
    test_data = DataIter(tok, test_data, test_label, batch_size=bsz, max_length=512)
    model.load(file_path + f'/{type_m}_{task}best_acc_')

    with torch.no_grad():
        model.eval()
        pre = []
        ref = []
        for dev_step, dev_batch in enumerate(test_data):
            output = model(input_ids=dev_batch[0]['input_ids'].to(device),
                           labels=dev_batch[1].to(device),
                           attention_mask=dev_batch[0]['attention_mask'].to(device))
            p = output.logits.argmax(dim=-1)
            pre += p.view(-1).tolist()
            ref += dev_batch[1].view(-1).tolist()
        sum_p = metric.compute(predictions=pre, references=ref)['accuracy']
    save_path = './save/few_shot/' + plm
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    report = f'data_size_{data_size}_seed_{seed}_task_{task}_p_{sum_p} \n'
    with open(save_path + f'/{type_m}_report.txt', 'a+', encoding='utf-8') as f:
        f.write(report)
    return sum_p
