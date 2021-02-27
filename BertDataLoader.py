# coding=utf-8
# bert mask model
#
import torch
import math
import random
import os
# from LAC import LAC


class DataSet(torch.utils.data.Dataset):
    def __init__(self, batch_size, types, config, train_ratio=0.8, base_file_path='data/all.txt'):
        assert types in ['train', 'inference']
        self.batch_size = batch_size
        self.config = config
        self.types = types
        self.base_file = base_file_path
        self.train_ratio = train_ratio
        self.max_len = 50
        if not os.path.exists('data/train.txt') or not os.path.exists('data/test.txt'):
            self.make_data(self.base_file)  # 构造数据、
        else:
            print('file exists !')
        if types == 'train':
            with open('data/train.txt', 'r', encoding='utf-8') as f1:
                self.data = [i.strip() for i in f1.readlines() if len(i.strip().split('#NLP#')[-1])<=self.max_len]
        else:
            with open('data/test.txt', 'r', encoding='utf-8') as f2:
                self.data = [i.strip() for i in f2.readlines() if len(i.strip().split('#NLP#')[-1])<=self.max_len]
        self.dataLen = len(self.data)
        print(types, 'DataLen :{}'.format(self.dataLen))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.dataLen

    def get_inform(self, types):  # 根据数据量和batch_size计算warm up的step 以及 总step
        assert types == 'train'
        max_epoch = self.config.max_epochs
        all_step = math.ceil(max_epoch / self.config.batch_size)   # 计算全部steps
        warm_up_step = 50
        return warm_up_step, all_step

    def make_data(self, base_file_paths):
        with open(base_file_paths, 'r', encoding='utf-8') as f:
            data_list = f.readlines()
        train_len = int(len(data_list) * self.train_ratio)
        # test_len = int(len(data_list) * (1-self.train_ratio))
        random.seed(0)
        random.shuffle(data_list)
        train_list = [i.strip()+'\n' for i in data_list[:train_len] if len(i.split('#NLP#')[-1]) >= 5]
        test_list_ = [i.strip()+'\n' for i in data_list[train_len:] if len(i.split('#NLP#')[-1]) >= 5]
        with open('data/train.txt', 'w', encoding='utf-8') as f1:
            f1.writelines(train_list)
        with open('data/test.txt', 'w', encoding='utf-8') as f2:
            f2.writelines(test_list_)
        del train_list, test_list_


def collect(data, splits='#NLP#'):
    train_d, labels, tags = [], [], []

    def send_data(data_s):
        train_d.append(data_s.split(splits)[-1])
        labels.append(int(data_s.split(splits)[1]))
        tags.append(1 if data_s.split(splits)[0] == 'new' else 0)
    [send_data(i) for i in data]
    return train_d, labels, tags

