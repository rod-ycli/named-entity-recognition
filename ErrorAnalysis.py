#!/usr/bin/env python
# coding: utf-8

# # Error Analysis
# 
# This file is to extract examples systematically for the error analysis. There are two main purposes using the following functions.
# 
# The first purpose is to compare how the predictions of the same entry changed between systems.
# User needs to specify which gold-pred pair to focus on and the number of examples wanted. They will be selected with an averaged stepsize.
# It will be useful to look into the effect of the features.
# 
# Then, the second purpose is to match the feature and the label in both train and dev set. It can be used to validate whether the feature in the training data affected the test results.

# In[1]:


import csv
from collections import Counter


# In[3]:


# Prediction comparison
def extract_wanted_entry(inputfile, gold, pred):
    wanted_entry = []
    with open(inputfile, 'r') as csvfile:
        content = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(content):
            if (row[-1]==pred) and (row[-2]==gold):
                row.insert(0, i)
                wanted_entry.append(row)
    return wanted_entry

def compare_basic_and_with_features(basic:list, wfeat:list, how_many=9999):
    basic_entry = extract_wanted_entry(basic[0], basic[1], basic[2])
    wfeat_entry = extract_wanted_entry(wfeat[0], wfeat[1], wfeat[2])
    
    basic_index = [entry[0] for entry in basic_entry]
    
    basic_eg = []
    wfeat_eg = []
    for entry in wfeat_entry:
        if entry[0] in basic_index:
            i = basic_index.index(entry[0])
            basic_eg.append(basic_entry[i])
            wfeat_eg.append(entry)
            
    every = max(len(basic_eg)//how_many, 1)

    print(f"""In total there are {len(basic_eg)} pairs of examples of the specified gold-pred comparison.
Here are {len(basic_eg[::every])} examples taken from every {every} pair(s).""")
    print("Basic system with only token:")
    for eg in basic_eg[::every]:
        print(eg)
    print()
    print("System with added features:")
    for eg in wfeat_eg[::every]:
        print(eg)
    print()
    return basic_eg, wfeat_eg


# In[4]:


# Finding out the feature that helped improve prediction
def find_the_frequent_feature_value(wfeat_eg, feat_list, feat_name):
    feat_index = feat_list.index(feat_name)
    feat_val = [eg[feat_index+1] for eg in wfeat_eg]
    count = sorted(Counter(feat_val).items(), key=lambda x:x[1], reverse=True)
    print("The count of the feature is:")
    print(count)
    print()
    return count

# Train-dev set matching
def match_feature_and_label(inputfile, feat_list, feat_name, feature, label):        
    feat_index = feat_list.index(feat_name)
    wanted_entry = []
    with open(inputfile, 'r') as csvfile:
        content = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(content):
            if (row[-1]==label) and (row[feat_index]==feature):
                row.insert(0, i)
                wanted_entry.append(row)
    return wanted_entry

def compare_train_and_dev_data(train:list, test:list, feat_list:list, how_many=9999):
    train_entry = match_feature_and_label(train[0], feat_list, train[1], train[2], train[3])
    test_entry = match_feature_and_label(test[0], feat_list, test[1], test[2], test[4])

    test_gold = test[3]    
    test_filter = [entry for entry in test_entry if (entry[-2]==test_gold)]
    
    every_train = max(len(train_entry)//how_many, 1)
    every_test = max(len(test_filter)//how_many, 1)

    print(f"""In total there are {len(train_entry)} examples of the specified feature and gold.
Here are {len(train_entry[::every_train])} taken from every {every_train} example(s).""")
    for eg in train_entry[::every_train]:
        print(eg)
    print()
    print(f"""In total there are {len(test_filter)} examples of the specified feature and gold-pred pair.
Here are {len(test_filter[::every_test])} taken from every {every_test} example(s).""")
    for eg in test_filter[::every_test]:
        print(eg)
    print()
    return train_entry, test_filter

def example_improved_by_feature(wfeat_eg:list, test_filter:list):
    results = [eg for eg in wfeat_eg if eg in test_filter]
    print(f"There are {len(results)} examples that are improved by feature.")
    print("The first 10 examples:")
    for result in results[:10]:
        print(result)
    print()
    return results


# In[5]:


feat_list = ['token', 'pattern', 'POS', 'prev_tok', 'next_tok', 'prev_pos', "next_pos"]


# In[6]:


# 1 PER-O > PER-PER
basic = ["../data/my_out.SVM.conll", "PER", "O"]
wfeat = ["../data/my_out_with_features.SVM.conll", "PER", "PER"]

basic_eg, wfeat_eg = compare_basic_and_with_features(basic, wfeat, 10)


# In[7]:


# 1 investigating POS

find_the_frequent_feature_value(wfeat_eg, feat_list, 'POS')

train = ["../data/Features_conll2003.train-preprocessed.conll","POS", "NNP", "PER"]
test = ["../data/my_out_with_features.SVM.conll","POS", "NNP", "PER","PER"]

train_eg, test_eg = compare_train_and_dev_data(train, test, feat_list, 5)
results = example_improved_by_feature(wfeat_eg, test_eg)


# In[8]:


# 2 ORG-O to ORG-ORG
basic = ["../data/my_out.SVM.conll", "ORG", "O"]
wfeat = ["../data/my_out_with_features.SVM.conll", "ORG", "ORG"]

basic_eg, wfeat_eg = compare_basic_and_with_features(basic,  wfeat, 15)


# In[9]:


#2 investigating the next_tok

find_the_frequent_feature_value(wfeat_eg, feat_list, 'next_tok')

train = ["../data/Features_conll2003.train-preprocessed.conll","next_tok", ",", "ORG"]
test = ["../data/my_out_with_features.SVM.conll", "next_tok", ",", "ORG", "ORG"]

train_eg, test_eg = compare_train_and_dev_data(train, test, feat_list, 5)
results = example_improved_by_feature(wfeat_eg, test_eg)


# In[ ]:




