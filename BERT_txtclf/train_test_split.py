#!/usr/bin/env python
# coding: utf-8

# # This program is to split a dataset into a training set and a testing set
# 
# Written on 2019/03/22 and modified on 2019/09/28 by Yuen-Hsien Tseng.

# In[1]:


# -*- coding: UTF-8 -*-
import time
time_Start = time.time()
from sklearn import model_selection
import pandas, numpy
import sys, os.path


# In[2]:


print("It takes %4.2f seconds to import packages."%(time.time()-time_Start))
print('''
Given a classificaiton file of a training set in 'class\\ttext\\n' format, 
    split the file (set) into a training set and a testing set.

  Usage:   python train_test_split.py Data_Name Train_File Train_Size_Ratio
  Example: python train_test_split.py joke TxtClf_Dataset/joke_All.txt 0.7

Note: the output is saved to files:
    'Data_Name/Data_Name_train.txt'
    'Data_Name/Data_Name_test.txt'

''')
print("sys.argv:", sys.argv)

if len(sys.argv) == 4 and sys.argv[1] != '-f':
    prog, Data_Name, data_file, Train_Size_Ratio = sys.argv
else:
    Data_Name = input("Enter the name of dataset:")
    data_file = input("Enter the dataset file: ")
    Train_Size_Ratio = input("Enter the train size ratio (default = 0.7): ")
Train_Size_Ratio = 0.7 if Train_Size_Ratio == '' else int(Train_Size_Ratio)
Test_Size_Ratio = 1 - Train_Size_Ratio


#data_file = 'PCWeb/PCWeb_utf8/PCWeb_All.txt'
#TrainSize = 1190

#data_file = 'PCNews/PCNews_utf8/PCNews_All.txt'
#TrainSize = 644

#data_file = 'CTC/CTC_utf8/CTC_All_sl.txt'
#TrainSize = 19901

#data_file = '20news-bydate/20news-bydate_All.txt'
#TrainSize = 11270

#data_file = 'Reuters/Reuters_All_sl.txt'
#TrainSize = 6561

#data_file = 'HiNet/HiNet_utf8/HiNet_All.txt'
#TrainSize = 232

#data_file = 'TxtClf_2.0/tweet2/training_random.txt'
#TrainSize = 4960

#data_file = 'twilio-sent-analysis-master/tweets_3_pn.txt'
#TrainSize = 1402

#data_file = 'joke/joke_all.txt'
#TrainSize = 2583
# Use next 2 lines (2019/03/22)
#data_file = '/Users/sam/GoogleDrive/AnacondaProjects/jokes/src/jokes.txt'
#TrainSize = 2390

print(f"data_file={data_file}, Train Ratio={Train_Size_Ratio}, Test Ratio={Test_Size_Ratio}")


# In[3]:



CharList, WordList = [], []

def Load_Data(file): # load the dataset
    labels, texts = [], []
    i = 0
    for line in open(file).read().split("\n"):
        if line == '': continue
        (label, text) = line.split("\t") # load my own data
        labels.append(label) # assume single label classification
        CharList.append(len(text))
        texts.append(text)
# create a dataframe using texts and lables
    DF = pandas.DataFrame()
    DF['text'] = texts
    DF['label'] = labels
    return DF

time_LoadData = time.time()

All_DF = Load_Data(data_file)

#CharList = [len(c) for c in All_DF['text']]
#WordList = [len(x.split()) for x in All_DF['text']]
TextCharsLen = max(CharList)
TextCharsAvg = sum(CharList) / len(CharList)

print("Max TextCharsLen =", TextCharsLen)
print("Avg Text Chars = %3d"%TextCharsAvg)

print("It takes %4.2f seconds to load, segment, and clean data."%(time.time()-time_LoadData))


# In[4]:


cat2num = pandas.Series(All_DF['label']).value_counts()
print(cat2num.sort_values(ascending=False))


# In[5]:


train_file = f'{Data_Name}/{Data_Name}_train.txt'
test_file = f'{Data_Name}/{Data_Name}_test.txt'
def Check_Folder_Files():
# https://tecadmin.net/python-check-file-directory-exists/
    if not os.path.exists(Data_Name):
        os.makedirs(Data_Name)
    if os.path.isfile(train_file):
        no = input(train_file +" already exists! Overwrite (Y/N)?")
        if no in ['', 'n', 'N', 'No', 'NO', 'no']:
            print("You choose not to overwrite. Program exit.")
            exit()
    if os.path.isfile(test_file):
        no = input(test_file + " already exists! Overwrite (Y/N)?")
        if no in ['', 'n', 'N', 'No', 'NO', 'no']:
            print("You choose not to overwrite. Program exit.")
            exit()
    return train_file, test_file


# In[6]:



def Save_Split_Train_Test(All_DF):
# https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    trainText_x, testText_x, train_yL, test_yL = model_selection.train_test_split(
    All_DF['text'], All_DF['label'], test_size=Test_Size_Ratio, stratify=All_DF['label'], random_state=42)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# https://stackoverflow.com/questions/34318141/zip-pandas-dataframes-into-a-new-dataframe
    #train_DF = pandas.concat([train_yL, trainText_x.str.split().str.join('')], axis=1)
    #test_DF = pandas.concat([test_yL, testText_x.str.split().str.join('')], axis=1)
    # Because of the original texts in Load_Data(), use next 2 lines rather than the above 2 lines
    train_DF = pandas.concat([train_yL, trainText_x], axis=1)
    test_DF = pandas.concat([test_yL, testText_x], axis=1)
    train_file, test_file = Check_Folder_Files()
# https://stackoverflow.com/questions/16923281/pandas-writing-dataframe-to-csv-file
    train_DF.to_csv(train_file, sep='\t', encoding='utf-8', index=False, header=False)
    test_DF.to_csv(test_file, sep='\t', encoding='utf-8', index=False, header=False)
    return trainText_x, testText_x, train_yL, test_yL


trainText_x, testText_x, train_yL, test_yL = Save_Split_Train_Test(All_DF)


# In[7]:


# See: https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups/38309823
print("train_yL: ", type(train_yL), ", shape:", train_yL.shape, ", unique:", train_yL.nunique(), "\n", train_yL.value_counts())
print("test_yL:  ", type(test_yL),  ", shape:", test_yL.shape, ", unique:", test_yL.nunique(), "\n", test_yL.value_counts())


# In[8]:


print(f"The output has been saved to files:\n'{train_file}'\n'{test_file}'")

