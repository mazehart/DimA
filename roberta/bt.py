import torch
from collections import OrderedDict
from utils.dataiter import DataIter
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, \
    get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.recorder import Recorder
from evaluate import load as load_metric
import os


def train_bt(task, seed, class_num, bsz, lr, epoch, plm, tag, rec_step):
    config = RobertaConfig.from_pretrained(plm)
    config.lr = lr
    config.epoch = epoch
    config.num_labels = class_num
    file_path = f'./save/{task}/{plm}/Bt_lr_{lr}_line_{epoch}_bsz_{bsz}_step_{rec_step}_seed_{seed}'
    os.makedirs(file_path) if not os.path.exists(file_path) else print('path exits!!!')
    torch.manual_seed(seed)
    device = f'cuda:{tag}' if torch.cuda.is_available() else 'cpu'
    model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=plm, config=config).to(device)
    tok = RobertaTokenizer.from_pretrained(plm)
    metric = load_metric('accuracy')
    train_data = torch.load(f'./data/{task}/train_data.pt')
    train_label = torch.load(f'./data/{task}/train_label.pt')
    dev_data = torch.load(f'./data/{task}/validation_data.pt')
    dev_label = torch.load(f'./data/{task}/validation_label.pt')
    train_data = DataIter(tok, train_data, labels=train_label, batch_size=bsz, max_length=128)
    dev_data = DataIter(tok, dev_data, labels=dev_label, batch_size=bsz, max_length=128)
    rec = Recorder(save_path=file_path + f'/Bt_{task}', record_steps=rec_step)
    train_param = [value for param, value in model.named_parameters() if 'bias' in param or 'class' in param]
    optim = AdamW(params=train_param, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=epoch * train_data.batch_num // 3,
                                                num_training_steps=epoch * train_data.batch_num)
    model.train()
    rec.start()
    for i in range(epoch):
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
                    sum_p = metric.compute(predictions=pre, references=ref)['accuracy']
                    sum_loss /= dev_data.batch_num
                    save_dir = rec.record_dev(train_step, sum_loss, sum_p)
                    for dirt in save_dir:
                        torch.save(OrderedDict({k:v for k, v in model.state_dict().items() if 'bias' in k or 'class' in k}), dirt + 'param.pt') if dirt is not None else None
                    rec.save_report()
                    print(rec.report)
                    print(scheduler.get_last_lr())
                    rec.rec_lr(scheduler.get_last_lr())
            model.train()
    time = rec.time()
    os.rename(file_path, file_path + f'_time_{time}')
    file_path = file_path + f'_time_{time}'
    test_data = torch.load(f'./data/{task}/test_data.pt')
    test_label = torch.load(f'./data/{task}/test_label.pt')
    test_data = DataIter(tok, test_data, labels=test_label, batch_size=bsz, max_length=512)
    save_path = './save/Bt_prediction/' + plm + f'/seed_{seed}'
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    dic = torch.load(file_path + f'/Bt_{task}best_acc_param.pt', map_location=device)
    model.load_state_dict(dic, strict=False)
    with torch.no_grad():
        model.eval()
        pre = []
        for test_step, test_batch in enumerate(test_data):
            output = model(input_ids=test_batch[0]['input_ids'].to(device),
                           labels=None,
                           attention_mask=test_batch[0]['attention_mask'].to(device))
            p = output.logits.argmax(dim=-1)
            pre += p.view(-1).tolist()
    torch.save(pre, save_path + f'/{task}_prediction.pt')
