# coding=utf-8
from BaseModel import *
from Config import *
from BertDataLoader import *
import pandas as pd
import time, os
import gc
import argparse, math


# from tqdm import tqdm


def read_test_excel(excel_path, select_plc_list=None, min_len=3, max_len=120):
    if excel_path.endswith('csv'):
        df = pd.read_csv(excel_path).fillna('')
    elif excel_path.endswith('xlsx'):
        df = pd.read_excel(excel_path).fillna('')
    else:
        print('wrong file format!,must be csv or xlsx')
        return
    print('select plc list', select_plc_list)
    wenan_list = []
    seq_list = []
    for index, seq in df.iterrows():
        if len(seq['desc'].strip()) < min_len or len(seq['desc'].strip()) > max_len:
            # print('too short :', seq['desc'])
            continue
        plc_code = seq['plc_code'].strip()
        if not select_plc_list:  # 如果 select_plc_list 是None
            seq_list.append(seq)
            wenan = seq['desc'].strip()
            wenan_list.append(wenan)
        else:
            if plc_code in select_plc_list:
                seq_list.append(seq)
                wenan = seq['desc'].strip()
                wenan_list.append(wenan)

    label_y = [str(0)] * len(wenan_list)
    df_new = pd.DataFrame(seq_list, columns=df.columns, )
    return label_y, wenan_list, df_new


def predict(mode='inference', threshold=0.5, input_name=None, output_name=None, min_len=3, max_len=130):
    configer = ConFig()
    test_set = DataSet(batch_size=configer.batch_size,
                       types='inference', train_ratio=configer.train_ratio,
                       config=configer, base_file_path='data/all.txt')
    time_start = time.time()

    base_bert_model_path = ''  # 这里随便设置，在预测阶段不影响
    have_bert_model = ClassBertModel(lr=configer.lr, bert_model_path=base_bert_model_path, mode=mode)
    predict_helper = BertModelHelper(lr=configer.lr,
                                     in_model=have_bert_model,
                                     mode=mode,
                                     train_set=test_set,
                                     test_set=test_set,
                                     batch_size=configer.batch_size)
    if torch.cuda.is_available():
        have_bert_model = have_bert_model.cuda()
        predict_helper = predict_helper.cuda()
    in_base = '/home/nlpbigdata/net_disk_project/yinwenbo/level3_in_data/pinjie_data/'
    # 加载csv数据，转换格式
    y, x, df = read_test_excel(in_base + input_name, min_len=min_len, max_len=max_len)

    print(x[:5])
    y_pred = []
    prob_0_list = []
    prob_1_list = []
    equal_list = []
    count = 0
    batch_size = 50

    block_num = math.ceil(len(x) / batch_size)
    for i_x in range(block_num):
        start = i_x * batch_size
        end = (i_x + 1) * batch_size
        # print(start, end)
        in_batch_data = x[start: end]
        in_batch_data = [i_ for i_ in in_batch_data]
        # print(in_batch_data[:10])
        count += 1
        # if len(x_one) < 3:
        #     print('==' * 4, i_x, x_one)
        pred_label, pred_logits = predict_helper.my_predict(in_batch_data)
        pred_label = pred_label.tolist()
        pred_logits = pred_logits.tolist()
        pre_label_list_tempo = [1 if pred_logits[0][1] >= threshold else 0]
        prob_0_list_temp = [round(i_s[0], 4) for i_s in pred_logits]
        prob_1_list_temp = [round(i_s[1], 4) for i_s in pred_logits]
        prob_0_list.extend(prob_0_list_temp)
        prob_1_list.extend(prob_1_list_temp)
        # equal_tempo = ['True' if pre_label_list_tempo[i_x] == y[i_x] else 'False' for i]
        # if pred_logits[0][1] >= threshold:
        #     pred_label = 1
        # else:
        #     pred_label = 0
        # prob_0_list.append(round(pred_logits[0][0], 4))
        # prob_1_list.append(round(pred_logits[0][1], 4))
        # equal = 'True' if pred_label == y[i_x] else 'False'
        # equal_list.append(equal)
        if count % 20 == 0:
            # print('{}/{}'.format(count, block_num, 'label:{}'.format(y[i_x]), 'predict:{}'.format(pred_label), in_batch_data))
            print('{}/{}'.format(count, block_num), 'predict:{} '.format(pred_label[0]), in_batch_data[0])
            gc.collect()
        y_pred.extend(pred_label)

    print("data pred ok!")
    print(len(x))
    print(len(prob_0_list), len(prob_1_list), len(y_pred))
    df.insert(loc=3, column='prob_0', value=prob_0_list)
    df.insert(loc=4, column='prob_1', value=prob_1_list)
    df.insert(loc=5, column='predict', value=y_pred)
    # df.insert(loc=7, column='equal', value=equal_list)
    print(df.columns)
    bath = '/home/nlpbigdata/net_disk_project/yinwenbo/level3_in_data/classed_data/'
    df.to_csv(bath + output_name, encoding='utf_8_sig', index=False)
    print("耗时:" + str(time.time() - time_start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', help="输入文件", type=int)
    parser.add_argument('--end_index', help="输入文件", type=int)
    parser.add_argument('--gpus', help="输入文件", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    if torch.cuda.is_available():
        print('using gpu ...')
    in_name = ['level3_{}_out.csv'.format(i) for i in range(args.start_index, args.end_index)]
    out_name = ['level1_{}_classed_out.csv'.format(i) for i in range(args.start_index, args.end_index)]
    # in_name = 'level3_0_out.csv'
    # out_name = 'level1_0_classed_out.csv'
    for in_name_s, out_name_s in zip(in_name, out_name):
        print('opening file {},  writing file {}'.format(in_name_s, out_name_s))
        predict(mode='inference', input_name=in_name_s, output_name=out_name_s)
