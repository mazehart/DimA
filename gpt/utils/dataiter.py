import torch
import random
random.seed(0)


def reindex(x, y):
    order = list(range(x.__len__()))
    random.shuffle(order)
    x_ = [x[t] for t in order]
    y_ = [y[t] for t in order]
    return x_, y_


def process(x, y):
    temp = -100 * torch.ones_like(x['input_ids'])
    x['input_ids'] = torch.cat((x['input_ids'], y['input_ids']), dim=-1)
    x['attention_mask'] = torch.cat((x['attention_mask'], y['attention_mask']), dim=-1)
    y['input_ids'] = torch.cat((temp, y['input_ids']), dim=-1)
    return x, y


class DataIter:

    def __init__(self, tok, sentences, labels, batch_size, num_data=None, max_length=256, test=False):
        self.tok = tok
        self.tok.pad_token = self.tok.eos_token
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.data_size = self.labels.__len__() if num_data is None or num_data > self.labels.__len__() else num_data
        self.batch_num = self.data_size // self.batch_size
        self.residual = True if self.data_size % self.batch_size else False
        self.index = 0
        self.max_length = max_length
        self.sentences = [x + "Summarization start:" for x in self.sentences]
        self.test = test

    def __next__(self):
        if self.index < self.batch_num:
            x = self.tok(self.sentences[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.tok(self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            self.index += 1
            if not self.test:
                x, y = process(x, y)
            return x, y

        elif self.index == self.batch_num and self.residual:
            x = self.tok(self.sentences[self.index * self.batch_size:], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.tok(self.labels[self.index * self.batch_size:], padding=True, max_length=256, return_tensors='pt', truncation=True)
            self.index += 1
            if not self.test:
                x, y = process(x, y)
            return x, y

        else:
            self.index = 0
            self.sentences, self.labels = reindex(self.sentences, self.labels)
            raise StopIteration

    def __iter__(self):
        return self

