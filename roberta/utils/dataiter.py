import torch
import random


random.seed(0)
def reindex(x, y):
    order = list(range(x.__len__()))
    random.shuffle(order)
    x_ = [x[t] for t in order]
    y_ = [y[t] for t in order]
    return x_, y_


class DataIter:

    def __init__(self, tok, sentences, labels, batch_size, num_data=None, max_length=128):
        self.tok = tok
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.data_size = self.labels.__len__() if num_data is None or num_data > self.labels.__len__() else num_data
        self.batch_num = self.data_size // self.batch_size
        self.residual = True if self.data_size % self.batch_size else False
        self.index = 0
        self.max_length = max_length

    def __next__(self):
        if self.index < self.batch_num:
            x = self.tok(self.sentences[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = torch.tensor(self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size])
            self.index += 1
            return x, y

        elif self.index == self.batch_num and self.residual:
            x = self.tok(self.sentences[self.index * self.batch_size:], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = torch.tensor(self.labels[self.index * self.batch_size:])
            self.index += 1
            return x, y

        else:
            self.index = 0
            self.sentences, self.labels = reindex(self.sentences, self.labels)
            raise StopIteration

    def __iter__(self):
        return self


class DataIterForGeneration:

    def __init__(self, tok, sentences, labels, batch_size, num_data=None, max_length=128):
        self.tok = tok
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.data_size = self.labels.__len__() if num_data is None or num_data > self.labels.__len__() else num_data
        self.batch_num = self.data_size // self.batch_size
        self.residual = True if self.data_size % self.batch_size else False
        self.index = 0
        self.max_length = max_length

    def __next__(self):
        if self.index < self.batch_num:
            x = self.tok(self.sentences[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.tok(self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            self.index += 1
            return x, y

        elif self.index == self.batch_num and self.residual:
            x = self.tok(self.sentences[self.index * self.batch_size:], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.tok(self.labels[self.index * self.batch_size:], padding=True, max_length=256, return_tensors='pt', truncation=True)
            self.index += 1
            return x, y

        else:
            self.index = 0
            self.sentences, self.labels = reindex(self.sentences, self.labels)
            raise StopIteration

    def __iter__(self):
        return self


class DataIterForGenerationTest:

    def __init__(self, tok, sentences, labels, batch_size, num_data=None, max_length=128):
        self.tok = tok
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.data_size = self.labels.__len__() if num_data is None or num_data > self.labels.__len__() else num_data
        self.batch_num = self.data_size // self.batch_size
        self.residual = True if self.data_size % self.batch_size else False
        self.index = 0
        self.max_length = max_length

    def __next__(self):
        if self.index < self.batch_num:
            x = self.tok(self.sentences[self.index * self.batch_size: (self.index + 1) * self.batch_size], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return x, y

        elif self.index == self.batch_num and self.residual:
            x = self.tok(self.sentences[self.index * self.batch_size:], padding=True, max_length=self.max_length, return_tensors='pt', truncation=True)
            y = self.labels[self.index * self.batch_size:]
            self.index += 1
            return x, y

        else:
            self.index = 0
            self.sentences, self.labels = reindex(self.sentences, self.labels)
            raise StopIteration

    def __iter__(self):
        return self


def tokenize_and_align_labels(tok, examples, example_label):
    tokenized_inputs = tok(
        examples,
        return_tensors='pt',
        padding=True,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(example_label):
        word_ids = [None]
        for j, word in enumerate(examples[i]):
            token = tok([word], add_special_tokens=False, is_split_into_words=True,)
            # print(token)
            word_ids += [j] * len(token['input_ids'])
        word_ids += [None]

        # word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            # elif word_idx != previous_word_idx:
            else:
                label_ids.append(label[word_idx])
                # label_ids.append(self.label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            # else:
            #     label_ids.append(-100)
            previous_word_idx = word_idx
        label_ids += [-100] * (tokenized_inputs['attention_mask'].size(-1) - label_ids.__len__())
        labels.append(label_ids)
    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs


class DataIterForToken:

    def __init__(self, tok, sentences, labels, batch_size, num_data=None, max_length=128):
        self.tok = tok
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.data_size = self.labels.__len__() if num_data is None or num_data > self.labels.__len__() else num_data
        self.batch_num = self.data_size // self.batch_size
        self.residual = True if self.data_size % self.batch_size else False
        self.index = 0
        self.max_length = max_length

    def __next__(self):
        if self.index < self.batch_num:
            x = tokenize_and_align_labels(self.tok, self.sentences[self.index * self.batch_size: (self.index + 1) * self.batch_size], self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size])
            # x = self.tok(, padding=True, max_length=self.max_length, return_tensors='pt', is_split_into_words=True,truncation=True)
            # y = torch.tensor(self.labels[self.index * self.batch_size: (self.index + 1) * self.batch_size])
            self.index += 1
            return x

        elif self.index == self.batch_num and self.residual:
            x = tokenize_and_align_labels(self.tok, self.sentences[self.index * self.batch_size: ], self.labels[self.index * self.batch_size: ])
            self.index += 1
            return x

        else:
            self.index = 0
            self.sentences, self.labels = reindex(self.sentences, self.labels)
            raise StopIteration

    def __iter__(self):
        return self
