import os 
import json 
flag =True
#print(len(os.listdir('train_videos')))  3451
#filenames =set(os.listdir(''))
#print(f"total movie:{len(filenames)}")
c={}
content = json.load(open('MSASL_train.json'))
content1 = json.load(open('MSASL_val.json'))
content2 = json.load(open('MSASL_test.json'))
for i in content2:
    gloss =i['text']
    if c.get(gloss)==None:
        c[gloss]=1
    else:
        c[gloss]+=1
        #print(i)
#print(c['door'])
c_sorted = sorted(c.items(), key = lambda x:(-(x[1]), x[0]))
max_10=10
for k ,v in c_sorted:
    if max_10>-1:print(k,v)
    max_10-=1
"""
train:
eat 57
nice 54
want 53
bird 50
orange 50
teacher 50
friend 48
like 48
what 48
white 48
fish 47
"""
"""
val :
nice 23
happy 22    
beautiful 20
big 20      
boy 19      
sorry 19
go 18
good 18
no 18
girl 17
like 17
"""
"""
test:

"""