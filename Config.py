# coding=utf-8


class ConFig:
    def __init__(self):
        self.lr = 0.00005
        # 选择bert参数更新的层级类型['low', 'middle', 'high', 'low_middle', 'middle_high', 'both']
        self.upgrade_type = 'high'
        batch_size = 8
        self.batch_size = batch_size
        self.generate_batch_size = batch_size
        self.base_model_path = 'Bert-wwm-ext/pytorch_model.bin'
        self.my_save_model_path = 'SavedModel/lm_last.pt'
        self.max_epochs = 10
        self.train_ratio = 0.01
        self.hidden_size = 768
        self.dropout = 0.4
        self.num_classes = 2
        self.num_filters = 10
        self.filter_sizes = [2, 4, 6]
        self.max_length = 50
