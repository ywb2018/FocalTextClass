
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('BaseBertModel/bert-base-chinese-vocab.txt')
# model = BertModel.from_pretrained('BaseBertModel/bert-base-chinese-pytorch_model.bin', return_dict=True)
strs = '坚 硬 抗冲击 ，密度大，抗冲击 性强，用 力压都不会变形'
inputs = tokenizer(strs, return_tensors="pt")
print(len(strs.replace(' ', '')), len(inputs['input_ids'][0]))
print(inputs['input_ids'][0])
print('dd')
# outputs = model(**inputs)
