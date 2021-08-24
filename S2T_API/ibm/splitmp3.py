
import json, os

with open('./set_dict.json') as json_file: 
    set = json.load(json_file)
    
# Create .shell script for audio splitting
file1 = open("split_mp3.sh","w") 
time = 1800 #seconds

for i in set['t']:
    case = i.split('.')[0]
    cmd = 'ffmpeg -i mp3/'+case+'.mp3 -f segment -segment_time '+str(time)+' -c copy mp3/split/'+case+'_%03d.mp3 \n'
    file1.write(cmd)
file1.close() 

print("split_mp3.sh created")

    
    