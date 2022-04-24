#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from typing import List, Dict


# In[2]:


def matching_tokens(conll1: List, conll2: List) -> bool:
    '''
    Check whether the tokens of two conll files are aligned
    
    :param conll1: tokens (or full annotations) from the first conll file
    :param conll2: tokens (or full annotations) from the first conll file
    
    :returns boolean indicating whether tokens match or not
    '''
    for i, row in enumerate(conll1):
        row2 = conll2[i]
        if row[0] != row2[0]:
            return False
    
    return True


# In[3]:


def read_in_conll_file(conll_file: str, delimiter: str = '\t'):
    '''
    Read in conll file and return structured object
    
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    
    :returns List of splitted rows included in conll file
    '''
    conll_rows = []
    with open(conll_file, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split(delimiter)
            if len(row) == 1:
                conll_rows.append([""]*rowlen)
            else:
                rowlen = len(row)
                conll_rows.append(row)
    return conll_rows


# In[4]:


def alignment_okay(conll1: str, conll2: str) -> bool:
    '''
    Read in two conll files and see if their tokens align
    '''
    my_first_conll = read_in_conll_file(conll1)
    my_second_conll = read_in_conll_file(conll2)
    
    return matching_tokens(my_first_conll, my_second_conll)


# In[5]:


def get_predefined_conversions(conversion_file: str) -> Dict:
    '''
    Read in file with predefined conversions and return structured object that maps old annotation to new annotation
    
    :param conversion_file: path to conversion file
    
    :returns object that maps old annotations to new ones
    '''
    conversion_dict = {}
    my_conversions = open(conversion_file, 'r')
    conversion_reader = csv.reader(my_conversions, delimiter='\t')
    for row in conversion_reader:
        conversion_dict[row[0]] = row[1]
    return conversion_dict


# In[6]:


def create_converted_output(conll_rows: List, annotation_identifier: int, conversions: Dict, outputfilename: str, delimiter: str = '\t'):
    '''
    Check which annotations need to be converted for the output to match and convert them
    
    :param conll_rows: rows with conll annotations
    :param annotation_identifier: indicator of how to find the annotations in the object (index)
    :param conversions: pointer to the conversions that apply. This can be external (e.g. a local file with conversions) or internal (e.g. prestructured dictionary). In case of an internal object, you probably want to add a function that creates this from a local file.
    
    '''
    with open(outputfilename, 'w') as outputfile:
        for row in conll_rows:
            annotation = row[annotation_identifier]
            if annotation in conversions:
                row[annotation_identifier] = conversions.get(annotation)
            if row[0] == "":
                outputfile.write("\n")
            else:
                outputfile.write(delimiter.join(row)+"\n")


# In[7]:


def preprocess_files(conll1: str, conll2: str, column_identifiers: List, conversions: Dict):
    '''
    Guides the full process of preprocessing files and outputs the modified files.
    
    :param conll1: path to the first conll input file
    :param conll2: path to the second conll input file
    :param column_identifiers: object providing the identifiers for target column
    :param conversions: path to a file that defines conversions
    '''
    if alignment_okay(conll1, conll2):
        conversions = get_predefined_conversions(conversions)
        my_first_conll = read_in_conll_file(conll1)
        my_second_conll = read_in_conll_file(conll2)
        create_converted_output(my_first_conll, column_identifiers[0], conversions, conll1.replace('.conll','-preprocessed.conll'))
        create_converted_output(my_second_conll, column_identifiers[1], conversions, conll2.replace('.conll','-preprocessed.conll'))
    else:
        print(conll1, conll2, 'do not align')

        
# Getting conll_ner_tags
read_gold = read_in_conll_file("../data/conll2003.dev.conll")

gold_ner_set = set()
for row in read_gold:
    gold_ner_set.add(row[3])
    
    
## Getting the full list of spacy NER tags
# The NER tags used are directly extracted from the file.
read_spacy = read_in_conll_file("../data/spacy_out.dev.conll")

spacy_ner_set = set()
for row in read_spacy:
    spacy_ner_set.add(row[2])
    
    
# Getting the full list of Stanford NER tags
# The NER tags used are directly extracted from the file.
read_stanford = read_in_conll_file("../data/stanford_out.dev.conll")

stanford_ner_set = set()
for row in read_stanford:
    stanford_ner_set.add(row[3])
        
        
# The conversion will include 3 components:
#   -  CoNLL tags: remove the BI labelling
#   -  spaCy tags: convert to CoNLL convention
#   -  Stanford tags: convert to CoNLL convention

# In[21]:

gold_conversion = set()
for item in gold_ner_set:
    if item == " ":
        continue  # Skipping the whitespaces
    if item.startswith("B-"):
        gold_conversion.add((item, item.lstrip('B-'))) # Each tuple consists of (tag with BIO marking -> tag)
    if item.startswith("I-"):
        gold_conversion.add((item, item.lstrip('I-')))


# In[24]:

spacy_conversion = set()
for item in spacy_ner_set:
    
    #{'', 'EVENT', 'PRODUCT', 'PERCENT', 'LAW', 'NORP', 'ORG', 'O', 'MONEY', 'LOC', 'DATE', 'LOCATION', 
    #'CARDINAL', 'TIME', 'FAC', 'LANGUAGE', 'ORDINAL', 'WORK_OF_ART', 'QUANTITY', 'PERSON', 'I-PER'}
    
    if item == "FAC" or item == "LOCATION":
        spacy_conversion.add((item, "LOC"))
    elif item == "PERSON" or item == "I-PER":
        spacy_conversion.add((item, "PER"))
    elif item == "O" or item == "ORG" or item == "LOC":
        continue  # 'O', 'ORG' and 'LOC' are the same with conll. No need for conversion. 
    elif item == "":
        spacy_conversion.add((item, "O")) # Adding notation "O" for tokens without an NER tag
    else:
        spacy_conversion.add((item, "MISC"))
        

# In[25]:

stanford_conversion = set()
for item in stanford_ner_set:
    
    # {'', 'COUNTRY', 'EMAIL', 'PERCENT', 'TITLE', 'RELIGION', 'STATE_OR_PROVINCE', 'CRIMINAL_CHARGE', 
    #'CAUSE_OF_DEATH', 'O', 'MONEY', 'ORGANIZATION', 'DATE', 'SET',  'TIME', 'LOCATION', 'DURATION',
    # 'CITY', 'ORDINAL', 'MISC', 'NUMBER', 'IDEOLOGY', 'NATIONALITY', 'PERSON'}
    
    # Tag 'SET' is part of SUTime. It applies to repeating events:  https://nlp.stanford.edu/software/corenlp-faq.shtml#set
    
    if item=="CITY" or item=="COUNTRY" or item=="LOCATION" or item=="STATE_OR_PROVINCE":
        stanford_conversion.add((item, "LOC"))
    elif item=="PERSON":
        stanford_conversion.add((item, "PER"))
    elif item=="ORGANIZATION":
        stanford_conversion.add((item, "ORG"))
    elif item=="O" or item=="MISC" or item=="":
        continue  # Don't do anything to "O"s, "MISC"s and empty lines
    else:
        stanford_conversion.add((item, "MISC"))

# In[27]:

# Join the three lists together as one set
conversion_all = gold_conversion.union(spacy_conversion, stanford_conversion)
print(conversion_all)

# Writing the conversion table
with open("./settings/conversions.tsv", 'w') as outputcsv:
    csvwriter = csv.writer(outputcsv, delimiter='\t')
    for conversion in conversion_all:
        print(conversion)
        csvwriter.writerow(conversion)


### Performing conversion and making the preprocessed data

# In[28]:

preprocess_files('../data/spacy_out.dev.conll','../data/conll2003.dev.conll', [2,3],'./settings/conversions.tsv')
preprocess_files('../data/stanford_out.dev.conll','../data/conll2003.dev.conll', [3,3],'./settings/conversions.tsv')
preprocess_files('../data/conll2003.train.conll','../data/conll2003.train.conll', [3,3],'./settings/conversions.tsv')