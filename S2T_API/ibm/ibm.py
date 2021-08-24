
import os

api_key = #set
lim = 5

check_lim = False
count = 0
file2 = open("IBM_API_Script.sh", "w")
for file in os.listdir('./wavs'):
    name = file.split('.')[0]
    pth = 'trans_segment/'+name+'_ibm_trans.json'
    if not os.path.exists(pth):
        cmd1 = 'curl -X POST -u \"apikey:'+api_key
        cmd2 = '\" --header \"Content-Type: audio/wav\" --data-binary @wavs/'+file
        cmd3=' \"https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/7efdb89d-7345-44e8-b67a-a39e31c2b0e2/v1/recognize?speaker_labels=true&audio_metrics=true\" > '+pth+' \n'
        file2.write(cmd1+cmd2+cmd3)
        count+=1
        if count>=lim and check_lim:
            print('Sample limit met')
            break
    
print('IBM API Processing Script written')