import torch
from utils.dataiter import DataIter
from methods.modeling_gpt2_lora import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils.recorder import Recorder
import os
from methods.count_param import count_param

def train_ad(task, seed, bsz, lr, epoch, plm, type_m, tag, rec_step):
    config = GPT2Config.from_pretrained(plm, cache_dir='/data1/zwx/ex_experiment/gpt/')
    config.type_m = type_m
    if type_m == 'lora':
        config.apply_lora = True
        config.apply_adapter = False
    else:
        config.apply_lora = False
        config.apply_adapter = True
    config.lora_attn_dim = 32
    config.lora_attn_alpha = 64
    config.lora_dropout = 0
    # config.adapter_type = 'pfeiffer'
    config.adapter_type = 'houlsby'
    config.adapter_size = 32
    config.epoch = epoch
    config.lr = lr
    
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
    
    param = [value for param, value in model.named_parameters() if type_m in param]
    optim = AdamW(params=param, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=epoch * train_data.batch_num // 3,
                                                num_training_steps=epoch * train_data.batch_num)
    model.train()
    param = count_param(model, param_name=type_m)
    file_path = f'./save/{task}/{plm}/{type_m}_lr_{lr}_line_{epoch}_bsz_{bsz}_step_{rec_step}_{param}'
    os.makedirs(file_path) if not os.path.exists(file_path) else print('path exits!!!')
    rec = Recorder(save_path=file_path + f'/{type_m}_{task}', record_steps=rec_step)
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
    test_data = DataIter(tok, test_data, test_label, batch_size=1, max_length=256, test=True)
    save_path = f'./save/{type_m}_prediction/' + plm
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    model.load(file_path + f'/{type_m}_{task}best_ec_')
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

    torch.save(pre, save_path + f'/{task}_prediction.pt')

if __name__ == '__main__':
    config = GPT2Config.from_pretrained("gpt2-large", cache_dir='/data1/zwx/ex_experiment/gpt/')
    config.apply_lora = True
    config.apply_adapter = False
    config.adapter_type = 'houlsby'
    config.lora_attn_dim = 16
    config.lora_attn_alpha = 32
    config.lora_dropout = 0
    torch.manual_seed(0)
    device = f'cuda:{2}' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", config=config, cache_dir='/data1/zwx/ex_experiment/gpt/').to(device)
    tok = GPT2Tokenizer.from_pretrained("gpt2-large", cache_dir='/data1/zwx/ex_experiment/gpt/')
    file_path = '/data1/zwx/ex_experiment/save/lora_prediction/gpt2-large/'
    test_data = torch.load(f'/data1/zwx/ex_experiment/data/xsum/test_data.pt')
    test_label = torch.load(f'/data1/zwx/ex_experiment/data/xsum/test_label.pt')
    test_data = DataIter(tok, test_data, test_label, batch_size=6, max_length=256, test=True)
    os.makedirs(file_path) if not os.path.exists(file_path) else None
    model.load('/data1/zwx/ex_experiment/save/xsum/gpt2-large/lora_lr_0.0005_line_1_bsz_16_step_150_time_53555.35925626755/lora_xsumbest_ec_')
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

    torch.save(pre, file_path + f'/xsum_prediction.pt')