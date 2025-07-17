import json
import time


class Recorder:

    def __init__(self, save_path, record_steps=200, best_score=1e4):

        self.save_path = save_path
        self.start_time = None
        self.epoch_start_time = []
        self.best_score = best_score
        self.best_score_time = ''
        self.report = ''
        self.score_list = []
        self.train_list = []
        self.acc_train_list = []
        self.acc_dev_list = []
        self.temp_train = 0
        self.record_steps = record_steps
        self.best_acc = 0
        self.acc = 0
        self.begin = time.time()
        self.rec_step = 0
        self.lr_list = []
        self.update_ec_list = []
        self.update_acc_list = []

    def start(self):

        self.start_time = time.ctime().replace(' ', '_').replace(':', '_')
        self.report = self.report + '=' * 10 + 'start train' + '=' * 10 + '\n' + 'start time:' + self.start_time + '\n'

    def epoch_start(self, idx):

        self.epoch_start_time.append(time.ctime().replace(' ', '_').replace(':', '_') + '_epoch_{}'.format(idx))
        self.report = self.report + '+' * 10 + 'epoch_{}'.format(idx) + ' starts' + '+' * 10 + '\n' + 'start time:' + self.epoch_start_time[-1] + '\n'

    def record_dev(self, train_step, loss, p):

        self.rec_step += 1

        self.score_list.append(loss.tolist())
        self.acc_dev_list.append(p)
        self.report = self.report + 'dev part: train_step_{}'.format(train_step) + '\n' + 'loss:{}'.format(loss) + '\n' + 'p:{}'.format(p) + '\n'
        save_dir1 = None
        save_dir2 = None
        save_dir3 = None
        if loss < self.best_score:
            self.best_score_time = self.epoch_start_time[-1] + '_train_step_{}_'.format(train_step)
            self.best_score = loss
            save_dir1 = self.save_path + 'best_ec_'
            self.update_ec_list.append(self.rec_step)
        if p > self.best_acc:
            self.best_score_time = self.epoch_start_time[-1] + '_train_step_{}_'.format(train_step)
            self.best_acc = p
            save_dir2 = self.save_path + 'best_acc_'
            self.update_acc_list.append(self.rec_step)
        save_dir3 = self.save_path + f'{train_step}' + f'loss={loss}'
        return save_dir1, save_dir2

    def save_report(self):

        with open(self.save_path + self.start_time + '_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.report, f)
        print('report saved!')

        with open(self.save_path + self.start_time + '_score.json', 'w', encoding='utf-8') as f:
            json.dump(self.score_list, f)
        print('score saved!')

        with open(self.save_path + self.start_time + '_train.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_list, f)
        print('train saved!')

        with open(self.save_path + self.start_time + '_acc_score.json', 'w', encoding='utf-8') as f:
            json.dump(self.acc_dev_list, f)
        print('score saved!')

        with open(self.save_path + self.start_time + '_acc_train.json', 'w', encoding='utf-8') as f:
            json.dump(self.acc_train_list, f)
        print('train saved!')

        with open(self.save_path + self.start_time + '_lr.json', 'w', encoding='utf-8') as f:
            json.dump(self.lr_list, f)
        print('train saved!')

        with open(self.save_path + self.start_time + '_acc_list.json', 'w', encoding='utf-8') as f:
            json.dump(self.update_acc_list, f)
        print('train saved!')

        with open(self.save_path + self.start_time + '_ec_list.json', 'w', encoding='utf-8') as f:
            json.dump(self.update_ec_list, f)
        print('train saved!')

    def record_train(self, train_step, loss, p):

        if train_step == 0:
            self.temp_train = 0
            self.acc = 0
        elif train_step % self.record_steps == 0:
            self.temp_train += loss.tolist()
            self.train_list.append(self.temp_train/self.record_steps)
            self.acc += p
            self.acc /= self.record_steps
            self.acc_train_list.append(self.acc)
            self.report = self.report + 'train part: train_step_{}'.format(train_step) + '\n' + 'loss:{}'.format(self.temp_train/self.record_steps) + '\n' + 'p:{}'.format(self.acc) + '\n'
            self.temp_train = 0
            self.acc = 0
        else:
            self.temp_train += loss.tolist()
            self.acc += p

    def time(self):
        return time.time() - self.begin

    def rec_lr(self, lr):
        self.lr_list.append(lr)
