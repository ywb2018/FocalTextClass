# coding=utf-8

from BaseModel import *
from Config import *
from BertDataLoader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(mode):
    configer = ConFig()
    train_set = DataSet(batch_size=configer.batch_size,
                        types='train', train_ratio=configer.train_ratio,
                        config=configer, base_file_path='data/all.txt')
    test_set = DataSet(batch_size=configer.batch_size,
                       types='inference', train_ratio=configer.train_ratio,
                       config=configer, base_file_path='data/all.txt')
    if mode == 'train':  # train
        classification_model = ClassBertModel(lr=configer.lr,
                                              bert_model_path=configer.base_model_path,
                                              upgrade_type='None',
                                              mode=mode, mask_rate=0.1)
        train_helper = BertModelHelper(lr=configer.lr,
                                       in_model=classification_model,
                                       mode=mode,
                                       train_set=train_set,
                                       test_set=test_set,
                                       batch_size=configer.batch_size)
        if torch.cuda.is_available():
            classification_model = classification_model.cuda()
            train_helper = train_helper.cuda()
        # train
        train_helper(show_step=60, valid_step=120)

    elif mode == 'inference':  # test
        base_bert_model_path = ''  # 这里随便设置，在预测阶段不影响
        have_bert_model = ClassBertModel(lr=configer.lr, bert_model_path=base_bert_model_path, mode=mode)
        predict_helper = BertModelHelper(lr=configer.lr,
                                         in_model=have_bert_model,
                                         mode=mode,
                                         train_set=train_set,
                                         test_set=test_set,
                                         batch_size=configer.batch_size)

        data_iter = predict_helper.get_my_data_loader('test')
        predict_out = []
        chinese_out = []
        for data in data_iter:
            predict_labels = predict_helper.my_predict(data)
            chinese_out.append(data)
            predict_out.append(predict_labels)


if __name__ == '__main__':
    types = ['train', 'inference']
    main(types[0])
