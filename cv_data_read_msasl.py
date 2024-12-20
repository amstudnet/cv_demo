import json 
import os
with open("MSASL_train.json") as path:
    x = json.load(path)
with open("MSASL_val.json") as path:
    x1 = json.load(path)
with open("MSASL_test.json") as path:
    x2 = json.load(path)
tar_x = len(os.listdir("train_videos"))
tar_x1 = len(os.listdir("val_videos"))
tar_x2 =len(os.listdir("test_videos"))
train_count,val_count,test_count = 0,0,0
glosses = []
for text in x:
    train_count+=1
    glosses.append(text['text'])
for text in x1:
    val_count+=1
    glosses.append(text['text'])
for text in x2:
    test_count+=1
    glosses.append(text['text']))
glosses=list(set(glosses))

print(f"total glosses:{len(glosses)}")#全部不重複的單字有幾個
print(f"total train:{train_count}")
print(f"total val:{val_count}")
print(f"total test:{test_count}")
print(f"total sample:{train_count+val_count+test_count}")
print(f"total train videos:{tar_x}")
print(f"total val videos:{tar_x1}")
print(f"total test videos:{tar_x2}")
"""
total glosses:1000 此程式計算有問題
total train:16054
total val:5287
total test:4172
total sample:25513
total train videos:3451
total val videos:679
total test videos:218
"""
