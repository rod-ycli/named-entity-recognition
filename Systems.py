#!/usr/bin/env python
# coding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import sys
import os
import gensim
import csv
import numpy as np

def extract_tokens_and_labels(trainingfile):
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
                targets.append(components[-1])
    return data, targets

def extract_more_features_and_labels(trainingfile):
    
    data = []
    targets = []
    all_tok = []
    all_pos = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            try:
                all_tok.append(components[0])
            except:
                all_tok.append("")
            try:
                all_pos.append(components[1])
            except:
                all_pos.append("")

    with open(trainingfile, 'r', encoding='utf8') as infile:
        for i, line in enumerate(infile):
            components = line.rstrip('\n').split()
            if len(components) > 0:
                
                # 1 token
                token = components[0]
                
                # 2 pattern
                pattern = []
                for char in components[0]:
                    if char.isupper():
                        pattern.append("A")
                    if char.islower():
                        pattern.append("a")
                    if char.isnumeric():
                        pattern.append("0")
                    if char.isalnum()==False:
                        pattern.append("-")
                pattern = "".join(pattern)
                
                # 3 POS
                pos = components[1]
                
                # 4 neighboring tokens
                prev_tok = all_tok[i-1]
                next_tok = all_tok[i+1]

                # 5 neighboring POS
                prev_pos = all_pos[i-1]
                next_pos = all_pos[i+1]

                feature_dict = {'token':token, 'pattern':pattern, 'POS':pos, 'prev_tok':prev_tok, 'next_tok':next_tok, 'prev_pos':prev_pos, 'next_pos':next_pos}
                data.append(feature_dict)
                                
                #gold is in the last column
                targets.append(components[-1])
                
    basename = os.path.basename(trainingfile)
    with open("../data/Features_"+basename, "w") as outfile:
        outfile.write("\t".join(feature_dict.keys())+"\tgold\n")
        for i, entry in enumerate(data):
            row = []
            for feature in entry.keys():
                row.append(entry[feature])
            row.append(targets[i])
            outfile.write("\t".join(row)+"\n")

    return data, targets


def extract_embeddings_as_features_and_gold(conllfile,word_embedding_model):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    '''

    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
            labels.append(row[-1])
    return features, labels


def extract_prev_current_and_next_embedding_as_features(data, word_embedding_model):
    '''
    Function that extracts features using word embeddings for preceding, current and the next token.
    Tokens are popped from the feature dictionary and later concatenated together with other features.
    
    :param data: a list of dictionary of features
    :param word_embedding_model: a pretrained word embedding model
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :return features: list of vector representation of tokens
    '''
    embedding = []
    features = []
    
    for entry in data:
        token = entry.pop('token')
        prev_tok = entry.pop('prev_tok')
        next_tok = entry.pop('next_tok')
        if token in word_embedding_model:
            vector1 = word_embedding_model[token]
        else:
            vector1 = [0]*300
        if prev_tok in word_embedding_model:
            vector2 = word_embedding_model[prev_tok]
        else:
            vector2 = [0]*300
        if next_tok in word_embedding_model:
            vector3 = word_embedding_model[next_tok]
        else:
            vector3 = [0]*300
        embedding.append(np.concatenate((vector1, vector2, vector3)))
        features.append(entry)
    return embedding, features


def create_classifier(train_features, train_targets, modelname):
   
    if modelname ==  'logreg':
        model = LogisticRegression(max_iter=10000)
    if modelname ==  'NB':
        model = ComplementNB()
    if modelname ==  'SVM':
        model = LinearSVC()
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model.fit(features_vectorized, train_targets)
    
    return model, vec

def create_classifier_with_embedding(train_features, train_targets, modelname):
   
    if modelname ==  'logreg':
        model = LogisticRegression(max_iter=10000)
    if modelname ==  'NB':
        model = ComplementNB()
    if modelname ==  'SVM':
        model = LinearSVC()
    vec = train_features
    model.fit(vec, train_targets)
    
    return model, vec

def create_classifier_mixed(embedding, features, train_targets):
    
    # Vectorizing sparse vectors
    vec = DictVectorizer()
    sparse_vectors = vec.fit_transform(features).toarray()
    
    # Combining dense and sparse vectors
    combined_vectors = []
    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,embedding[index]))
        combined_vectors.append(combined_vector)
    
    # Creating and training an SVM model
    model = LinearSVC(max_iter=10000)
    model.fit(combined_vectors, train_targets)
    
    return model, vec


def classify_data(model, vec, inputdata, outputfile):
    features, gold_test = extract_tokens_and_labels(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    
def classify_data_with_features(model, vec, inputdata, outputfile):
    features, gold_test = extract_more_features_and_labels(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    inputdata = os.path.split(inputdata)
    for i, line in enumerate(open(inputdata[0]+"/Features_"+inputdata[1], 'r')):
        if i > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def classify_with_embedding(model, word_embedding_model, inputdata, outputfile):
  
    features = []
    conllinput = open(inputdata, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        if len(row) > 3:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
    
    predictions = model.predict(features)
    
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):

        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()


def classify_with_mixed(model, vec, word_embedding_model, inputdata, outputfile):
  
    data, targets = extract_more_features_and_labels(inputdata)
    embedding, features = extract_prev_current_and_next_embedding_as_features(data, word_embedding_model)
    
    # Transforming sparse vectors
    sparse_vectors = vec.transform(features).toarray()
    
    # Combining dense and sparse vectors
    combined_vectors = []
    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,embedding[index]))
        combined_vectors.append(combined_vector)
    
    # Make predictions
    predictions = model.predict(combined_vectors)
    
    outfile = open(outputfile, 'w')
    counter = 0
    inputdata = os.path.split(inputdata)
    for i, line in enumerate(open(inputdata[0]+"/Features_"+inputdata[1], 'r')):
        if i > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    inputfile = argv[2]
    outputfile = argv[3]
    
    # Basic systems with only tokens
    training_features, gold_labels = extract_tokens_and_labels(trainingfile)
    for modelname in ['logreg', 'NB', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll', f'.{modelname}.conll'))
                      
    # With more features
    training_features, gold_labels = extract_more_features_and_labels(trainingfile)
    for modelname in ['logreg', 'NB', 'SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data_with_features(ml_model, vec, inputfile, outputfile.replace('.conll', f'_features.{modelname}.conll'))
                      
    # With embedding

    language_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)

    training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
    for modelname in ['SVM']:
        ml_model, vec = create_classifier_with_embedding(training_features, gold_labels, modelname)
        classify_with_embedding(ml_model, language_model, inputfile, outputfile.replace('.conll','_emb.SVM.conll'))
                      
    # Dense and sparse vectors
    training_features, gold_labels = extract_more_features_and_labels(trainingfile)
    embedding, feature = extract_prev_current_and_next_embedding_as_features(training_features, language_model)
    ml_model, vec = create_classifier_mixed(embedding, feature, gold_labels)
    classify_with_mixed(ml_model, vec, language_model, inputfile, outputfile.replace('.conll','_mixed.SVM.conll'))
                      
########

trainingfile = "../data/conll2003.train-preprocessed.conll"
inputfile = "../data/conll2003.dev-preprocessed.conll"
outputfile = "../data/my_out.conll"

arg = ['placeholder', trainingfile, inputfile, outputfile, ]

if __name__ == '__main__':
#        main(arg)  ## Uncomment this line and comment the remaining lines to run with the default arguments
    main()
