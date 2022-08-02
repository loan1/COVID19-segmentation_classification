#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

#tao duong dan den data
train_path = '../dataset/train/'
test_path = '../dataset/test/'

#doc file metadata
train_metadata = '../dataset/' + 'train.txt'
test_metadata = '../dataset/' + 'test.txt'
train_txt= pd.read_csv(train_metadata, sep=" ", header=None)
test_txt = pd.read_csv(test_metadata, sep=" ", header=None)

#gan ten cot
train_txt.columns= ["patient id","file_name","class","source"]

#https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
train_txt['label'] = np.where(train_txt['class']== 'negative', 0, 1) #tao cot label theo dk cua class

#chi lay cot filename va label (1,4)
train_txt1 = train_txt.iloc[:,[1,4]]

#gan ten cot tren file test
test_txt.columns= ["patient id","file_name","class","source"]

test_txt['label'] = np.where(test_txt['class']== 'negative', 0, 1) #tao cot label theo dk cua class

test_txt1 = test_txt.iloc[:,[1,4]]

test_txt1.to_csv('../dataset/t.csv',index=False) 

#https://www.freecodecamp.org/news/python-write-to-file-open-read-append-and-other-file-handling-functions-explained/
with open('../dataset/test_set.txt', 'w') as f:
    f.write(
        test_txt1.to_string(header = True, index = False)
    )
f.close()

train_set,val_set=train_test_split(train_txt1,test_size=0.2, random_state = 42, shuffle=True)

with open('../dataset/train_set.txt', 'w') as f:
    f.write(
        train_set.to_string(header = True, index = False)
    )
f.close()

with open('../dataset/val_set.txt', 'w') as f:
    f.write(
        val_set.to_string(header = True, index = False)
    )
f.close()



