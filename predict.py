import torch
import time

from models import ERNIE
from pytorch_pretrained import BertTokenizer
import models
from importlib import import_module
from datetime import datetime
import ast
import csv

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def readClassList(classFile):
    systemClass = []
    with open(classFile, 'r', encoding='UTF-8') as f:
        for line in f:
            systemClass.append(line)
    return systemClass

def readClassDict(classFile):
    companyClass = {}
    with open(classFile, 'r', encoding='UTF-8') as f:
        for line in f:
            companyClass[line.split('	')[0]] = line.split('	')[1]
    return companyClass

class use_model():
    def __init__(self):
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.systemClass = readClassList('Dataset/class.txt')
        self.model_name = 'ERNIE'
        self.x = import_module('models.' + self.model_name)
        self.config = self.x.Config('Dataset')
        self.model = ERNIE.Model(self.config)
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.to(self.config.device)


    def fastPredict(self, content, pad_size):
        start_time = time.time()

        content

        tokenized_text = self.tokenizer.tokenize(content)
        tokenized_text = [CLS] + tokenized_text
        seq_len = len(tokenized_text)
        mask = []

        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        if pad_size:
            if len(tokenized_text) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(tokenized_text))
                token_ids += ([0] * (pad_size - len(tokenized_text)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size

        x = torch.LongTensor([token_ids]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([seq_len]).to(self.device)
        mask = torch.LongTensor([mask]).to(self.device)

        in_data = (x, seq_len, mask)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(in_data)

        #print(outputs)
        predicted_label = torch.argmax(outputs, dim=1).item()
        end_time = time.time()
        print("Time usage:", end_time - start_time)
        return self.systemClass[predicted_label]


        

if __name__ == '__main__':

    model_handler = use_model()
    usrIn = input("请输入文本：")
    result=model_handler.fastPredict(usrIn,64)
    print(result)