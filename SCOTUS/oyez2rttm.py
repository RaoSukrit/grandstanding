import os, sys, json


label = 20 # unknown speaker label counter (leave room for 20 judges)
process_sd=True
spkr_dict = dict()
cnt=0
trans_path = './oyez/'
rttm_path = './rttm/'

for filename in os.listdir(trans_path):
    case_name = filename.split('.')[0]
    #check if it has been ran successfully
    if os.path.exists(rttm_path+case_name+'.rttm'):
      print('Case', case_name, 'already processed')
      continue
    else:
      print('Processing Case', case_name)
      f = open(trans_path+filename,'r')
      k = f.readlines()
      f.close()

      timelst = []
      for u in k:
        t0, t1, spkr = u.split(' ')[0:3]
        #build speaker dictionary
        if spkr[-14:]=='scotus_justice':
            if spkr not in spkr_dict:
              spkr_dict[spkr] = cnt
              cnt+=1
              if cnt>=20:
                print("ERROR: NEED MORE JUDGE ROOM")
        else:
            if spkr not in spkr_dict:
              spkr_dict[spkr] = label
              label+=1       
        timelst.append((float(t0),float(t1),spkr))
        
      torttm = []
      for i, event in enumerate(timelst):
        torttm.append(' '.join(['SPEAKER '+case_name+' 1', str(event[0]), str(round(event[1]-event[0], 2)), '<NA> <NA>', event[2],'<NA> <NA>']))
        
      with open(rttm_path+case_name+'.rttm', 'w') as filehandle:
          for listitem in torttm:
              filehandle.write('%s\n' % listitem)
if process_sd:
  print('New Spkr Dict Saved')
  with open('./spkr_dict.json', 'w') as outfile:  
    json.dump(spkr_dict, outfile) 

