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
    
    
model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=1)
model.load_state_dict(torch.load(notes_path+'BERTforGS'))

model.eval()
model.to(device)

diar = 'oyez'
api_list = ['OYEZ', 'GOOG', 'IBM']
in_path = notes_path + 'toBERT_'+diar+'.json'
with open(in_path) as f:
  to_BERT = json.load(f)
  
  
to_model = {}
drop_info = {}
for api in api_list:
  drop_temp = []
  test_gs = []
  test_info = []
  for key in to_BERT.keys():
    case = to_BERT[key][api]
    for u in case:
      spkr=u[0]
      text=u[1].replace('%HESITATION', '') #IBM cleaning
      text = [i for i in text.split(' ') if i!='--'] # Oyez Cleaning

      # performs example dropping
      if len(text)>=min_words: #word threshold
        test_gs.append(' '.join(text))
        test_info.append((key, spkr, u[2][0], u[2][1]))
      else:
        drop_temp.append((key, spkr, u[2][0], u[2][1]))
  drop_info[api] = drop_temp
  to_model[api] = [test_gs, test_info]
  
  
item_tab = []
for key in to_BERT.keys():
  case = to_BERT[key]['OYEZ']
  for u in case:
    item_tab.append((key, spkr, u[2][0], u[2][1]))
    
    
# processing scotus examples
for test_api in api_list:
  out_path = notes_path + 'BERT4GS_SCOTUS_'+api+'_'+diar+'.csv'
  test = to_model[test_api][0]
  test_enc = tokenizer(test, truncation=True, padding=True)
  test_dataset = GrandStand_Dataset(test_enc, [0 for i in test])

  predicted_grandstanding_scores = []
  for batch in test_dataset:
    with torch.no_grad():
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      outputs = model(input_ids.reshape(1,-1), attention_mask=attention_mask.reshape(1,-1))
      predicted_grandstanding_scores.append(outputs.logits[0].item())
  
  SCOTUS = pd.DataFrame(to_model[test_api][1], columns=['Case', 'Spkr', 'Start', 'End'])
  SCOTUS['Speech'] = test
  SCOTUS['BERT-GS_Scores']=predicted_grandstanding_scores
  SCOTUS.to_csv(out_path)