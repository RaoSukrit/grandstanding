'''
Training BERTForSequenceClassification Model to estimate the amount of grandstanding in judicial speech. Data from 'When Do Politicians Grandstand'(https://www.journals.uchicago.edu/doi/abs/10.1086/709147?af=R). This makes use of the transformers library from https://huggingface.co/.

'''

import torch, csv, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
notes_path = 'drive/MyDrive/Colab Notebooks/BERTforGrandStanding/' 
bert_model = 'bert-base-uncased'
final_model_save_name = None # original Name was BERTforGS, changed to prevent overwrite

data = pd.read_csv(notes_path+'gs_data.csv') # from aforementioned paper

#loads r package stopwords that are saved down manually
with open(notes_path+'stpl.csv', newline='') as f:
    reader = csv.reader(f)
    stopwords = list(reader)
stopwords = [i[1] for i in stopwords]
stopwords = stopwords + ['ain', 'aren', 'can', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'im', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
train, test = train_test_split(data, test_size=.2, random_state=42, shuffle=True)
train, val = train_test_split(train, test_size=.2, random_state=42, shuffle=True)


desc = test['sentimentit_score'].describe(percentiles=[.1,.25,.5,.75,.9])

hi = desc['90%']
lo = desc['10%']
mhi = desc['75%']
mlo = desc['25%']


test_hard = test[(test['sentimentit_score']>hi) | (test['sentimentit_score']<lo)]
test_med = test[(test['sentimentit_score']>mhi) | (test['sentimentit_score']<mlo)]

x_tr = [i for i in train['speech']]
x_val = [i for i in val['speech']]
x_te = [i for i in test['speech']]
y_te =  [i for i in test['sentimentit_score']]

xm_te = [i for i in test_med['speech']]
ym_te =  [i for i in test_med['sentimentit_score']]

xh_te = [i for i in test_hard['speech']]
yh_te =  [i for i in test_hard['sentimentit_score']]

print(len(train), len(val), len(test), len(test_med), len(test_hard))




#from huggingface documentation
class GrandStand_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
        
        
tokenizer = BertTokenizer.from_pretrained(bert_model)
train_encodings = tokenizer(x_tr, truncation=True, padding=True)
val_encodings = tokenizer(x_val, truncation=True, padding=True)
test_encodings = tokenizer(x_te, truncation=True, padding=True)

train_dataset = GrandStand_Dataset(train_encodings, [i for i in train['sentimentit_score']])
val_dataset = GrandStand_Dataset(val_encodings, [i for i in val['sentimentit_score']])
test_dataset = GrandStand_Dataset(test_encodings, y_te)


test_encodings_med = tokenizer(xm_te, truncation=True, padding=True)
test_dataset_med = GrandStand_Dataset(test_encodings_med, ym_te)

test_encodings_hard = tokenizer(xh_te, truncation=True, padding=True)
test_dataset_hard = GrandStand_Dataset(test_encodings_hard, yh_te)



model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=1)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    weight_decay=0.1,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    eval_steps = 80,
    logging_steps = 80,
    evaluation_strategy='steps',
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,   # the instantiated 🤗 Transformers model to be trained
    tokenizer=tokenizer,                        
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset           
)

trainer.train()

trainer.evaluate()

print('RMSE Full Test Set BERT:', np.sqrt(trainer.predict(test_dataset)[2]['test_loss']))

# SAVE TRAINED MODEL
torch.save(model.state_dict(), notes_path+final_model_save_name)