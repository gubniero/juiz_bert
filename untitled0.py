pip install datasets

pip install transformers

pip install torch

from os import path, mkdir
from shutil import rmtree
import os
import glob

from IPython.display import clear_output
from time import sleep

from pyarrow import Table
from pyarrow import parquet

import pandas as pd
import torch

from transformers import AutoTokenizer, BertForSequenceClassification, AdamW

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

class BertTextClassifier(torch.nn.Module):
  def __init__(self, num_labels):
    super(BertTextClassifier, self).__init__()
    self.num_labels = num_labels
    self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',
                                                                num_labels=num_labels)

  def forward(self, labels, input_ids, token_type_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
    logits = outputs.logits
    return logits

#definitions

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_pretrain = BertTextClassifier(3).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

optimizer = torch.optim.SGD(model_pretrain.parameters(), lr=1e-4, momentum=0.5)
#optimizer = AdamW(model_pretrain.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
  
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

from datasets import load_dataset

num_labels = 3


def transform_dataset(example):
  label_map = {"no": 0, "partial": 1, "yes": 2}
  
  text = example['ementa_text'].lower()
  label = example['judgment_label']
  label = label_map[label]
  #print(label)
  tokens = tokenizer(text, padding='max_length', truncation=True, max_length=256)
  example = {**example, **tokens}
  target = torch.zeros(num_labels)
  target[label] = 1
  example['labels'] = target.tolist()
  #print(example)
  return example

dataset = load_dataset("joelito/brazilian_court_decisions")
dataset = dataset.map(transform_dataset)
dataset.set_format('torch')
dataset = dataset.remove_columns(['process_number', 'orgao_julgador', 'publish_date', 'judge_relator', 'decision_description', 'ementa_text', 'judgment_label', 'judgment_text', 'unanimity_text', 'unanimity_label'])
train_dataloader = DataLoader(dataset['train'].select(range(3000)), shuffle=True, batch_size=16)
validation_dataloader = DataLoader(dataset['validation'], shuffle=True, batch_size=16)
test_dataloader = DataLoader(dataset['test'], shuffle=True, batch_size=1)

sample = next(iter(train_dataloader))
  sample = {k:v.to(DEVICE) for k,v in sample.items()}

  with torch.no_grad():
    output = model_pretrain(**sample)
  #del sample
  torch.cuda.empty_cache()
  print()

def train_model(model,
                data_loader,
                criterion,
                optimizer,
                lr_scheduler=None,
                device=torch.device('cpu')):
  model.train()
  mean_loss = 0
        
  for batch in data_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model(**batch)
    loss = criterion(output, batch['labels'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    mean_loss += loss.item()
        
  mean_loss /= len(data_loader)

  if lr_scheduler is not None:
    lr_scheduler.step()
        
  return mean_loss

def eval_epoch(model,
               data_loader,
               criterion,
               device):
  model.eval()
  correct = 0
  total = 0
  loss = 0
  num_samples = len(data_loader)

  with torch.no_grad():
    for batch in data_loader:
      batch = batch
      batch = {k: v.to(device) for k, v in batch.items()}
      labels = torch.argmax(batch['labels'], 1) 
      outputs = model(**batch)
                
      predicted = torch.argmax(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      loss += criterion(outputs, labels).item()

    accuracy = 100 * correct / total
    loss = loss/num_samples

    return {'loss': loss, 'acc':accuracy}

num_epochs = 60
#model_pretrain = BertTextClassifier(3).to(DEVICE)
train_loss3 = []
eval3 = []

for epoch in range(num_epochs):
  
  
  train_epoch = train_model(model_pretrain,
                            train_dataloader,
                            criterion,
                            optimizer,
                            #lr_scheduler,
                            device=DEVICE)
  clear_output(wait=True)
  print('treinamento da epoca: ', epoch)
  print('loss do treinamento: ', train_epoch)
  
  eval_epoch_ = eval_epoch(model_pretrain,
                           validation_dataloader,
                           criterion,
                           DEVICE)
  print('validação da epoca: ',eval_epoch_)
  
      

  train_loss3.append(train_epoch)

  eval3.append(eval_epoch_)
  print('\n\n')
  

  #if eval_epoch_['acc'] <


check_model = eval_epoch(model_pretrain,
                         test_dataloader,
                         criterion,
                         DEVICE)

eval3 = pd.DataFrame(eval3)

eval3.to_csv('eval3.csv')

train_loss3 = pd.DataFrame(train_loss3)

train_loss3.to_csv('train_loss3.csv')

print(check_model)
