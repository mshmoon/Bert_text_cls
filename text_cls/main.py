# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer,BertModel
from data_preprocess import process_data 
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Linear(768, 200)

    def forward(self, x):
        context = x["input_ids"]  # 输入的句子
        mask = x["attention_mask"]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x["token_type_ids"] 

        output = self.bert(context, attention_mask=mask,token_type_ids=token_type_ids)
        pooled = output.pooler_output
        out = self.fc(pooled)

        return out


class My_Dataset(Dataset):

    def __init__(self, content,label):
        self.content = content 
        self.label = label
        self.tokenizer =  BertTokenizer.from_pretrained("bert-base-chinese")
        assert len(self.label) == len(self.content)

    def __getitem__(self, index):
        text = self.content[index]
        label = float(self.label[index])
        inputs = self.tokenizer(text,max_length = 512,pad_to_max_length = True, truncation=True, return_tensors='pt') 
        inputs = {k:v.squeeze() for k,v in inputs.items()}
        label = torch.tensor(label,dtype = torch.long)
        return inputs,label 
    def __len__(self):
        return len(self.label)


# 权重初始化，默认xavier
def train(model, dataset):
    model.train()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    dataloader = DataLoader(dataset,batch_size = 12)
    
    for epoch in range(20):
        for i, (inputs, labels) in enumerate(dataloader):   # trains, labels ==>  (x, seq_len, mask), y
            inputs = {k:v.cuda() for k,v in inputs.items()}
            labels = labels.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch,i,loss)


if __name__ == "__main__":
    label,content = process_data("dataset/train.json")
    dataset = My_Dataset(content,label)     
    model = Model()
    train(model,dataset)
  


