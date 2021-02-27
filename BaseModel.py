# coding=utf-8
# bert mask model
#
# focal loss 知乎
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Config import *
from MyLoss import FocalLoss
from BertDataLoader import *
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, BertConfig, AdamW
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score


class TextCNN(nn.Module):
    def __init__(self):
        self.config = ConFig()
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.num_filters, (k, self.config.hidden_size)) for k in self.config.filter_sizes])
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.num_filters * len(self.config.filter_sizes), self.config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, alpha=0.6, gamma=1.6, focus=2, use_focalloss=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.use_focalloss = use_focalloss
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.textcnn_classifier = TextCNN()
        if use_focalloss:
            self.loss_func = FocalLoss(alpha=alpha, gamma=gamma, focus=focus)
        else:
            self.loss_func = CrossEntropyLoss()

    def forward(
            self,
            tags=None,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        # bert 输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.textcnn_classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # 计算损失
        if tags is not None and self.use_focalloss:  # 如果tags不为None
            loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1), tags)
        elif labels is not None and self.use_focalloss:
            loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = outputs[0].new_zeros(1)[0]
        outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class ClassBertModel(nn.Module):  # classify bert model,用于自训练   # bert model 更新参数的类型
    def __init__(self, lr, bert_model_path, upgrade_type='both', mode='train', mask_rate=0.4,
                 bert_config_path='Bert-wwm-ext/bert_config.json',
                 bert_vocab_path='Bert-wwm-ext/vocab.txt'
                 ):
        assert upgrade_type.lower() in ['emb', 'low', 'middle', 'high', 'low_middle', 'middle_high', 'both', 'none']
        assert mode in ['train', 'inference']
        super().__init__()
        self.mask_rate = mask_rate
        self.learning_rate = lr
        # 根据设置 选择哪些层可以进行更新
        self.upgrade_layer_dict = {
            'none': [None],
            'low': ['0'],
            'middle': ['5'],
            'high': ['10', '11'],
            'low_middle': ['0', '5'],
            'middle_high': ['5', '11'],
            'both': ['0', '5', '11']
        }
        self.bert_model_path = bert_model_path
        self.bert_vocab_path = bert_vocab_path
        self.bert_config_path = bert_config_path
        if mode == 'train':
            self.tokenizer, self.model = self._build_bert_train(self.upgrade_layer_dict[upgrade_type.lower()])
        else:
            self.tokenizer, self.model = self._build_have_bert()

    def _build_bert_train(self, upgrade_layer_list):  # build model，同时选定部分层进行参数更新
        def compare(string, list_in):
            compare_result = sum([1 if i in string else 0 for i in list_in])  # 如果能匹配上，则和不为0
            return compare_result > 0

        # 构建tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_path)
        config = BertConfig.from_json_file(self.bert_config_path)
        # 构建mask bert model
        print('building bert model, path: {}'.format(self.bert_model_path))
        # model = self.tfs.BertForSequenceClassification.from_pretrained(self.bert_model_path,
        #                                                                from_tf=False, config=config)
        model = MyBertForSequenceClassification.from_pretrained(self.bert_model_path,
                                                                from_tf=False, config=config)
        # upgrade_layer_list 选择 某些层进行更新
        if None in upgrade_layer_list:  # 如果不更新层，则直接返回模型
            return tokenizer, model
        aim_list = ['bert.encoder.layer.{}'.format(i) for i in upgrade_layer_list]
        for name, p in model.named_parameters():
            if compare(name, aim_list):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return tokenizer, model

    def _build_have_bert(self):
        base_configs = BertConfig.from_json_file(self.bert_config_path)
        tokenizer = BertTokenizer.from_pretrained(self.bert_vocab_path)
        print('building bert model, path: {}'.format(self.bert_model_path))
        # bert_model_shell = self.tfs.BertForSequenceClassification(config=base_configs)
        bert_model_shell = MyBertForSequenceClassification(config=base_configs)  # 警告也没关系
        return tokenizer, bert_model_shell

    def forward(self, x, y=None, tags=None, do_mask=False):
        # assert isinstance(x, list), '输入x必须是list类型!'
        batch_tokens = self.tokenizer(x, padding=True, return_tensors='pt')  # padding, pt_tensor
        if do_mask:
            attention_mask = batch_tokens['attention_mask']
            for ind, mask_ in enumerate(attention_mask):
                select = torch.tensor(
                    random.sample(range(1, mask_.sum() - 1), int(mask_.sum() * self.mask_rate))).long()
                # mask_.index_fill_(1, select, 0)
                mask_.view(1, -1).index_fill_(1, select, 0).view(-1)
                attention_mask[ind] = mask_
            # print(attention_mask)
            batch_tokens['attention_mask'] = attention_mask
        gpu_batch_tokens = {}
        if torch.cuda.is_available():
            # 把x放入gpu
            for key in batch_tokens:
                gpu_batch_tokens[key] = batch_tokens[key].cuda()
                # print(key, gpu_batch_tokens[key].device)
        else:
            gpu_batch_tokens = batch_tokens
            # 把y放入x对应的gpu
        del batch_tokens
        # print(gpu_batch_tokens['attention_mask'].device)
        # print(gpu_batch_tokens['input_ids'].device)
        if y:
            y = torch.tensor(y, device=list(gpu_batch_tokens.values())[0].device).long()
        if tags is not None:
            tags = torch.tensor(tags, device=list(gpu_batch_tokens.values())[0].device).long()
        # print(y.device, tags.device)
        class_loss, logits = self.model(**gpu_batch_tokens, labels=y, tags=tags)
        del gpu_batch_tokens, y, tags
        return class_loss, logits


class BertModelHelper(nn.Module):
    # 训练、预测、保存、加载
    def __init__(self, lr, in_model, train_set, test_set, decay_gamma=0.8,
                 batch_size=16, mode='train', model_path='SavedModel/'):
        assert mode in ['train', 'inference']
        super().__init__()
        self.configers = ConFig()
        self.learning_rate = lr
        self.model_path = model_path
        self.mode = mode
        self.batch_size = batch_size
        # 定义模型
        if self.mode == 'inference':
            print('loading my model ...')
            self.model = self.load_lm(in_model, path=self.configers.my_save_model_path)
        else:  # train
            self.model = in_model

        self.train_d_set = train_set  # train data set
        # self.valid_d_set = valid_set  # valid data set
        self.test_d_set = test_set  # test data set
        self.train_data_iter = torch.utils.data.DataLoader(self.train_d_set, self.batch_size,
                                                           shuffle=True, collate_fn=collect)  # train iter
        # self.valid_data_iter = torch.utils.data.DataLoader(self.valid_d_set, self.batch_size,
        #                                                    shuffle=True)  # valid iter
        test_batch = self.test_d_set.dataLen
        self.test_data_iter = torch.utils.data.DataLoader(self.test_d_set, test_batch,
                                                          shuffle=True, collate_fn=collect)  # valid iter
        self.num_warm_up_step, self.training_step = self.train_d_set.get_inform(types='train')
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.learning_rate)
        # 动态调整的学习率
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_gamma)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def forward(self, show_step=20, valid_step=20):  # 训练bert-class 语言模型
        # self.model.train()
        for epoch in range(1, self.configers.max_epochs + 1):
            steps = 0
            train_hard_samples = []
            valid_hard_samples = []
            pass
            train_bad_loss_all, valid_bad_loss_all = 0, 0
            for train_batch, train_label_batch, train_tags in tqdm(self.train_data_iter):  # train数据迭代
                batch_loss, batch_logits = self.model(train_batch, train_label_batch, train_tags, do_mask=False)
                # 从每个batch中找出最难分的数据,最后写入文件
                train_bad_word, train_bad_loss = self.get_worst_data(epoch, steps, train_batch, train_label_batch,
                                                                     batch_logits)
                train_hard_samples.append(train_bad_word)
                train_bad_loss_all += float(train_bad_loss)
                self.optimizer.zero_grad()  # 清空梯度
                batch_loss.backward()  # 反向传播
                self.optimizer.step()  # 梯度更新

                if steps % show_step == 0 and steps:
                    loss = float(batch_loss.item())
                    print('training|| epoch:{} - step: {} - loss:{}'.
                          format(epoch, steps, round(float(loss), 4)))

                if steps % valid_step == 0 and steps:
                    with torch.no_grad():
                        for valid_batch, valid_label_batch, valid_tags in self.test_data_iter:  # valid数据迭代
                            batch_valid_loss, batch_valid_logit = self.model(valid_batch, valid_label_batch, valid_tags)
                            self.make_metrics(batch_valid_logit, valid_label_batch)
                            valid_bad_word, valid_bad_loss = self.get_worst_data(epoch, steps, valid_batch,
                                                                                 valid_label_batch, batch_valid_logit)

                            valid_hard_samples.append(valid_bad_word)
                            valid_bad_loss_all += float(valid_bad_loss)

                            print('validating|| epoch:{} - step: {} - loss:{}'.format(
                                epoch, steps, round(float(batch_valid_loss.item()), 4)),
                                self.optimizer.state_dict()['param_groups'][0]['lr'])
                            break

                steps += 1
                # 保存模型
                if steps % 20 == 0:
                    # print(torch.cuda.memory_summary(device=batch_loss.device))
                    # self.scheduler.step(batch_loss.item())  # 学习率衰减
                    torch.cuda.empty_cache()
            self.scheduler.step(epoch)
            self.save_lm(self.model, epoch, path_base='SavedModel/')
            with open('logs/{}_train_hard_samples.txt'.format(epoch), 'a+', encoding='utf-8') as f_1:
                f_1.writelines(train_hard_samples)
            with open('logs/{}_valid_hard_samples.txt'.format(epoch), 'a+', encoding='utf-8') as f_2:
                f_2.writelines(valid_hard_samples)
            print('\nepoch - {}, train_bad_loss_all - {}, valid_bad_loss_all - {}\n'.format(
                epoch, train_bad_loss_all, valid_bad_loss_all
            ))
            # print('end epoch||  LR:{}'.format(round(self.optimizer.state_dict()['param_groups'][0]['lr'],4)))

    def get_my_data_loader(self, types):
        if types == 'inference':
            return self.test_data_iter
        else:
            return self.train_data_iter

    def my_predict(self, x):
        loss_, predict_logits = self.model(x)  # 初步断定为BertForMaskedLM 输出了预测分数，待定
        # print(predict_logits)
        soft_max_out = F.softmax(predict_logits, dim=-1).cpu().data
        out = torch.argmax(soft_max_out, dim=-1)
        out = out.cpu().data  # 放到cpu上
        return out, soft_max_out

    @staticmethod
    def get_worst_data(epoch, steps, data_chinese, label_batch, predict_logits_tensors):
        # 根据logit_predict 结果找出难分的样本，写入文件
        softmax_out = F.softmax(predict_logits_tensors, dim=-1)
        if torch.cuda.is_available():
            label_batch = torch.tensor(label_batch, device=softmax_out.device)
        score_list = []
        for soft_, label_ in zip(softmax_out, label_batch):
            # print(soft_.device)
            loss_ = F.cross_entropy(soft_.view(1, -1), torch.tensor(label_).view(-1))
            score_list.append(float(loss_.detach()))
        index, max_loss = np.argmax(score_list), np.max(score_list)
        label = float(label_batch[index])
        head = '##'.join([str(epoch), str(steps), str(round(max_loss, 2)), str(label)])
        return head + '#NLP#' + data_chinese[index] + '\n', max_loss

    @staticmethod
    def save_lm(model, epoch, path_base=None):  # 模型保存
        assert path_base is not None, 'path can not be None !'
        model_save_path = path_base + 'lm_{}.pt'.format(epoch)
        model_save_path2 = path_base + 'lm_last.pt'

        torch.save(model.state_dict(), model_save_path)
        torch.save(model.state_dict(), model_save_path2)  # 最新一次语言模型
        print('saved')

    def load_lm(self, model, path=''):  # 模型加载
        model_save_path = path if len(path) else self.model_path
        print('test ||  loading parameters ...')
        param_dict = torch.load(model_save_path, map_location='cpu')
        try:
            model.load_state_dict(param_dict)
        except Exception as e:
            raise e
        return model

    @staticmethod
    def make_metrics(logit, labels):
        # labels = labels.cpu()
        predicts = torch.argmax(logit, dim=-1).cpu()
        precision = precision_score(y_true=labels, y_pred=predicts)
        recall = recall_score(y_true=labels, y_pred=predicts)
        f1_scores = f1_score(y_true=labels, y_pred=predicts)
        report = classification_report(y_true=labels, y_pred=predicts)
        print('precision - {}, recall - {}, f1 - {}'.format(precision, recall, f1_scores))
        print(report)
        # return precision, recall, f1_scores, report
