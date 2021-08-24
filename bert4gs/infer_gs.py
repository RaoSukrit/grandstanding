import torch, csv, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import describe

notes_path = 'drive/MyDrive/Colab Notebooks/BERTforGrandStanding/'
set_path = notes_path+'set_dict.json'
trans_path = notes_path+'trans/'


gold_path = notes_path+'BERT4GS_SCOTUS_GOLD.csv'
bert_model = 'bert-base-uncased'
device='cuda' #check runtime
min_words = 40
perc = [.1,.25,.5,.75,.9]


from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(bert_model)

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
        
        
        
# for evaluating scoring results
def eval_df(df,score_thresh=1, verbose=True):
  size = []
  thresh = []
  for c in df['Case'].unique():
    temp = df[df['Case']==c]
    seg_size = temp['End']-temp['Start']
    score = temp['BERT-GS_Scores']
    size.append(len(seg_size))
    thresh.append([s for s in score if s>score_thresh])

  thresh = [j for i in thresh for j in i]
  if verbose:
    print("# segs per case:", np.mean(size))
    print("# segs with GS >",score_thresh,":", len(thresh))
  return size, thresh
  
with open(set_path) as f:
    set_dict = json.load(f)
    
    
# create list of judicial segments   
gs = []
info=[]

for filename in set_dict['t']:
    case_name = filename.split('.')[0]
    #print('Processing Case', case_name)
    f = open(trans_path+case_name+'.txt','r')
    k = f.readlines()
    f.close()
    for u in k:
      temp = u.split('scotus_justice')
      if len(temp)>1: #only judicial speech
        t0, t1, spkr = temp[0].split(' ')
        text = temp[1]
        text = [i for i in text.split(' ') if i!='--']
        if len(text)>=min_words: #word threshold
          gs.append(' '.join(text))
          info.append((case_name, spkr, t0, t1))
lab = [0 for i in gs] #fake labels for dataset class


print("# Cases:", len(set_dict['t']))
print("# Speech Segments:", len(gs))

SCOTUS = pd.DataFrame(info, columns=['Case', 'Spkr', 'Start', 'End'])
SCOTUS['Speech'] = gs

# Infer GrandStanding and Save Results
scotus_enc = tokenizer(gs, truncation=True, padding=True)
scotus_dataset = GrandStand_Dataset(scotus_enc, lab)

model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=1)

model.load_state_dict(torch.load(notes_path+'BERTforGS'))

model.eval()
model.to(device)
predicted_grandstanding_scores = []

for batch in scotus_dataset:
  with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model(input_ids.reshape(1,-1), attention_mask=attention_mask.reshape(1,-1))
    predicted_grandstanding_scores.append(outputs.logits[0].item())
    
    
SCOTUS['BERT-GS_Scores']=predicted_grandstanding_scores

SCOTUS['BERT-GS_Scores'].describe(percentiles=[.1,.25,.5,.75,.9])

_,_ = eval_df(df, 0)
print('--')
_,_ = eval_df(df, 1)

SCOTUS.to_csv(gold_path)


