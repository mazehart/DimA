import torch
from utils.dataiter import DataIter
from methods.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.recorder import Recorder
import os
from methods.count_param import count_param

def train_aug(task, seed, bsz, lr, epoch, plm, add_dim, tag, rec_step):
    config = GPT2Config.from_pretrained(plm)
    config.use_bias = False
    config.aug_dim = add_dim * config.n_head
    config.apply_aug_mlp = True
    config.apply_aug_att = True
    config.apply_aug_v = True
    config.apply_aug_o = True
    config.line_m = 1
    
    torch.manual_seed(seed)
    device = f'cuda:{tag}' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained(plm, config=config, cache_dir='/data1/zwx/ex_experiment/gpt/').to(device)
    tok = GPT2Tokenizer.from_pretrained(plm, cache_dir='/data1/zwx/ex_experiment/gpt/')
    train_data = torch.load(f'./data/{task}/train_data.pt')
    train_label = torch.load(f'./data/{task}/train_label.pt')
    dev_data = torch.load(f'./data/{task}/validation_data.pt')
    dev_label = torch.load(f'./data/{task}/validation_label.pt')
    train_data = DataIter(tok, train_data, train_label, batch_size=bsz, max_length=256)
    dev_data = DataIter(tok, dev_data, dev_label, batch_size=bsz, max_length=256)
    
    aug_param = [value for param, value in model.named_parameters() if 'aug' in param]
    optim = AdamW(params=aug_param, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=epoch * train_data.batch_num // 3,
                                                num_training_steps=epoch * train_data.batch_num)
    model.train()
    param = count_param(model, param_name='aug')
    file_path = f'./save/{task}/{plm}/Augment_dim_{add_dim}_lr_{lr}_{param}'
    os.makedirs(file_path) if not os.path.exists(file_path) else print('path exits!!!')
    rec = Recorder(save_path=file_path + f'/Augment_{task}', record_steps=rec_step)
    rec.start()
    for i in range(epoch):
        rec.epoch_start(i)
        for train_step, batch in enumerate(train_data):
            output = model(input_ids=batch[0]['input_ids'].to(device),
                           labels=batch[1]['input_ids'].to(device),
                           attention_mask=batch[0]['attention_mask'].to(device))
            loss = output.loss
            loss.backward()
            optim.step()
            model.zero_grad()
            scheduler.step()
            if train_step % rec.record_steps == 0 and train_step != 0:
                with torch.no_grad():
                    model.eval()
                    sum_loss = 0
                    pre = []
                    ref = []
                    for dev_step, dev_batch in enumerate(dev_data):
                        output = model(input_ids=dev_batch[0]['input_ids'].to(device),
                                       labels=dev_batch[1]['input_ids'].to(device),
                                       attention_mask=dev_batch[0]['attention_mask'].to(device))
                        loss = output.loss
                        p = output.logits.argmax(dim=-1)
                        sum_loss += loss
                    sum_loss /= dev_data.batch_num
                    save_dir = rec.record_dev(train_step, sum_loss, 0)
                    for dirt in save_dir:
                        model.save(dirt) if dirt is not None else None
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
    test_data = DataIter(tok, test_data, test_label, batch_size=12, max_length=256, test=True)
    save_path = './save/Aug_prediction/' + plm
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    model.load(file_path + f'/Augment_{task}best_ec_')
    with torch.no_grad():
        model.eval()
        pre = []
        for test_step, test_batch in enumerate(test_data):
            output = model.generate(input_ids=test_batch[0]['input_ids'].to(device),
                                    attention_mask=test_batch[0]['attention_mask'].to(device),
                                    max_length=512,
                                    num_beams=5,
                                    early_stopping=True, )
            pre += tok.batch_decode(output[:, test_batch[0]['input_ids'].size(-1):], skip_special_tokens=True)

    torch.save(pre,
               save_path + f'/{task}_{add_dim}_prediction.pt')

if __name__ == '__main__':
    add_dim = 9
    plm = "gpt2-large"
    task = 'xsum'
    config = GPT2Config.from_pretrained(plm)
    config.use_bias = False
    config.aug_dim = add_dim * config.n_head
    config.apply_aug_mlp = True
    config.apply_aug_att = True
    config.apply_aug_v = True
    config.apply_aug_o = True
    config.line_m = 1
    torch.manual_seed(0)
    device = f'cuda:{4}' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained(plm, config=config, cache_dir='/data1/zwx/ex_experiment/gpt/').to(device)
    tok = GPT2Tokenizer.from_pretrained(plm, cache_dir='/data1/zwx/ex_experiment/gpt/')
    test_data = torch.load(f'/data1/zwx/ex_experiment/data/{task}/test_data.pt')
    test_label = torch.load(f'/data1/zwx/ex_experiment/data/{task}/test_label.pt')
    test_data = DataIter(tok, test_data, test_label, batch_size=8, max_length=256, test=True)
    save_path = '/data1/zwx/ex_experiment/gpt/save/Aug_prediction/' + plm
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    model.load('/data1/zwx/ex_experiment/save/xsum/gpt2-large/Augment_dim_9_lr_0.0002_time_80696.61816930771/Augment_xsumbest_ec_')
    with torch.no_grad():
        model.eval()
        pre = []
        for test_step, test_batch in enumerate(test_data):
            output = model.generate(input_ids=test_batch[0]['input_ids'].to(device),
                                    attention_mask=test_batch[0]['attention_mask'].to(device),
                                    max_length=512,
                                    num_beams=5,
                                    early_stopping=True, )
            pre += tok.batch_decode(output[:, test_batch[0]['input_ids'].size(-1):], skip_special_tokens=True)
            torch.save(pre,
               save_path + f'/{task}_{add_dim}_prediction.pt')
