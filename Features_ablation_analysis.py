#!/usr/bin/env python
# coding: utf-8

# In[29]:


import sys
import pandas as pd
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
import gensim
import csv
import Results_and_basic_evaluation as RBE


# In[2]:


# Evaluation metrixes


# In[39]:


def print_evaluation(predictions, goldlabels, model, selected_feature):
    counts = RBE.obtain_counts(goldlabels, predictions)
    RBE.provide_confusion_matrix(counts)
    evaluation = {}
    evaluation[(model, selected_feature)] = RBE.calculate_precision_recall_fscore(counts)
    RBE.provide_output_tables(evaluation)


# In[4]:


# Taking different features


# In[37]:


def extract_features_and_gold_labels(conllfile, selected_features, feature_to_index):
    '''Function that extracts features and gold label from preprocessed conll (here: tokens only).
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    
    features = []
    labels = []
    conllinput = open(conllfile, 'r')

    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    ### Taken from https://stackoverflow.com/questions/14257373/skip-the-headers-when-editing-a-csv-file-using-python, 10 Dec 2021
    next(csvreader, None)  # skip the headers
    ###
    for row in csvreader:
        feature_value = {}
        for feature_name in selected_features:
            row_index = feature_to_index.get(feature_name)
            feature_value[feature_name] = row[row_index]
        features.append(feature_value)
        labels.append(row[-1])
    return features, labels


def create_vectorizer_and_classifier(features, labels, modelname):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a logistic regression classifier
    
    :param features: feature-value pairs
    :param labels: gold labels
    :type features: a list of dictionaries
    :type labels: a list of strings
    
    :return lr_classifier: a trained LogisticRegression classifier
    :return vec: a DictVectorizer to which the feature values are fitted. 
    '''
    vec = DictVectorizer()
    tokens_vectorized = vec.fit_transform(features)   
    if modelname ==  'logreg':
        lr_classifier = LogisticRegression(max_iter=10000)
    if modelname ==  'NB':
        lr_classifier = ComplementNB()
    if modelname ==  'SVM':
        lr_classifier = LinearSVC()
    lr_classifier.fit(tokens_vectorized, labels)
    
    return lr_classifier, vec


def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features, feature_to_index):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    
    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: 
    
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    features, goldlabels = extract_features_and_gold_labels(testfile, selected_features, feature_to_index)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)
    
    return predictions, goldlabels


# In[35]:


def main(argv):

    trainingfile = argv[1]
    testfile = argv[2]
    feature_to_index = argv[3]
    models = argv[4]

    features = list(feature_to_index.keys())
    
    # for each pair of feature
    feature_pairs = [(features[0], features[i]) for i in range(1, len(features))]
    for pair in feature_pairs:
        print('Extracting Features:', pair)
        feature_values, labels = extract_features_and_gold_labels(trainfile, pair, feature_to_index)
        
        # for each model
        for modelname in models:
            print('Training classifier:', modelname)
            lr_classifier, vec = create_vectorizer_and_classifier(feature_values, labels, modelname)
            print('Evaluation:', modelname, pair)
            predictions, goldlabels = get_predicted_and_gold_labels(testfile, vec, lr_classifier, pair, feature_to_index)
            print_evaluation(predictions, goldlabels, modelname, pair)


# In[ ]:


# Setting some variables that we will use multiple times
# names must match key names of dictionary feature_to_index
# the functions with multiple features and analysis

trainfile = '../data/Features_conll2003.train-preprocessed.conll'
testfile = '../data/Features_conll2003.dev-preprocessed.conll'
feature_to_index = {'token': 0, 'pattern': 1, 'POS': 2, 'prev_tok': 3, 'next_tok': 4, 'prev_pos':5, 'next_pos':6}
models = ['logreg', 'NB', 'SVM']

argv = ['placeholder', trainfile, testfile, feature_to_index, models]

main(argv)



