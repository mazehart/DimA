import torch
from utils.dataiter import DataIter
from prefix_encoder import GPT2PrefixModel
from transformers import GPT2Tokenizer, GPT2Config
import os


def test(lr, plm, tag, task='xsum', seed=0):
    torch.manual_seed(seed)
    device = f'cuda:{tag}' if torch.cuda.is_available() else 'cpu'
    config = GPT2Config.from_pretrained(plm)
    config.pre_seq_len = 40
    config.prefix_hidden_size = 512
    config.prefix_projection = True
    config.hidden_size = config.n_embd
    config.num_hidden_layers = config.n_layer
    config.plm = plm
    model = GPT2PrefixModel(config=config).to(device)
    tok = GPT2Tokenizer.from_pretrained(plm)
    test_data = torch.load(f'../data/{task}/test_data.pt')
    test_label = torch.load(f'../data/{task}/test_label.pt')
    test_data = DataIter(tok, test_data, test_label, batch_size=3, max_length=256, test=True)
    save_path = './save/Pt_prediction/' + plm
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    dic = torch.load(lr)
    model.load_state_dict(dic)
    with torch.no_grad():
        model.eval()
        pre = []
        for test_step, test_batch in enumerate(test_data):
            output = model.generate(input_ids=test_batch[0]['input_ids'].to(device),
                                    attention_mask=test_batch[0]['attention_mask'].to(device),
                                    max_length=512,
                                    num_beams=5,
                                    early_stopping='.')
            pre += tok.batch_decode(output[:, test_batch[0]['input_ids'].size(-1):], skip_special_tokens=True)

    torch.save(pre, save_path + f'/{task}_prediction.pt')


if __name__ == '__main__':
    lr = '/home/mqh/ex_experiment/save/xsum/gpt2-medium/Pt_lr_0.0003_line_1_bsz_8_step_150_time_20615.078463554382/Pt_xsumbest_ec_param.pt'
    plm = 'gpt2-medium'
    test(lr, plm, tag='0')
