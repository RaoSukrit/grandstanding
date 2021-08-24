
import os, json

bucket = 'scotus_test/'

save_path = "C:\jt_work\Research\Grand\data\google_trans"

with open('./set_dict.json') as j:
    set = json.load(j)
       
count = 0
file2 = open("google_SDK_Script.bat", "w")

for file in set['t']:
    name = file.split('.')[0]
    pth = save_path+name+'.json'
    if not os.path.exists(pth):
        cmd1 = 'python GoogleSpeech.py gs://'+bucket+name+'.flac'
        cmd2 = 'move '+name+'.json '+save_path
        file2.write(cmd1+'\n'+cmd2+'\n')
        count+=1
    
print("Google API Processing Script written, ", count, " files expected to be processed")