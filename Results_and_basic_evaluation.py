#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
from collections import defaultdict, Counter


# In[2]:


def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    if inputfile.endswith(".conll"):
        conll_input = pd.read_csv(inputfile, sep=delimiter, on_bad_lines='skip', quotechar=delimiter, header=None)
        annotations = conll_input[int(annotationcolumn)].tolist()
    else:
        conll_input = pd.read_csv(inputfile, sep=delimiter, quotechar=delimiter)
        annotations = conll_input[annotationcolumn].tolist()
    return annotations


# In[4]:


def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    allannotations = sorted(list(zip(machineannotations, goldannotations)))

    evaluation_counts = defaultdict(Counter)
    for gold_key in set(goldannotations):
        for gold_value in set(goldannotations):
            evaluation_counts[gold_key][gold_value] = 0
    for pred, gold in allannotations:
        evaluation_counts[pred][gold] += 1
    return evaluation_counts


# In[6]:


def get_contingency_table(evaluation_counts, pred):
    '''
    Provides an overview of true positives, false positives and false negatives for each class.
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns a dictionary with the number of true positives, false positives and false negatives for each class.
    '''
    tp = evaluation_counts[pred][pred]
    fp = sum(evaluation_counts[pred].values())-tp
    fn = 0
    for other_pred in evaluation_counts:
        if other_pred!=pred:
            fn+=evaluation_counts[other_pred][pred]
    return tp, fp, fn
    
def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''
    eval_scores = {}
    tp_all = []
    fp_all = []
    fn_all = []
    prec_all = []
    recall_all = []
    f1_all = []
    for pred in evaluation_counts:
        tp, fp, fn = get_contingency_table(evaluation_counts, pred)
        if (tp+fp==0):
            precision = 0
        else:
            precision = tp/(tp+fp)
        if (tp+fn==0):
            recall = 0
        else:
            recall = tp/(tp+fn)
        if precision==0 and recall==0:
            f_score=0
        else:
            f_score = (2*precision*recall)/(precision+recall)
        eval_scores[pred] = {}
        eval_scores[pred]['precision'] = "%.3f" % (precision)
        eval_scores[pred]['recall'] = "%.3f" % (recall)
        eval_scores[pred]['f-score'] = "%.3f" % f_score
        
        tp_all.append(tp)
        fp_all.append(fp)
        fn_all.append(fn)
        prec_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f_score)
        
    ### macro avg, micro avg
    macro_precision = sum(prec_all)/len(prec_all)
    macro_recall = sum(recall_all)/len(recall_all)
    macro_f1 = sum(f1_all)/len(f1_all)
    
    micro_precision = sum(tp_all)/(sum(tp_all)+sum(fp_all))
    micro_recall = sum(tp_all)/(sum(tp_all)+sum(fn_all))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)
    
    eval_scores["macro avg"] = {'precision': "%.3f" % (macro_precision),
                               'recall': "%.3f" % (macro_recall),
                               'f-score': "%.3f" % (macro_f1)}
    eval_scores["micro avg"] = {'precision': "%.3f" % (micro_precision),
                               'recall': "%.3f" % (micro_recall),
                               'f-score': "%.3f" % (micro_f1)}

    return eval_scores


# In[7]:


def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''

    confusion_matrix = pd.DataFrame.from_dict(evaluation_counts)
    print(confusion_matrix)


# In[8]:


def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome


# In[9]:


def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)


def create_system_information(system_information):
    '''
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.
    
    :param system_information is the input as from a commandline or an input file
    '''
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list
    

# In[10]:


def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param goldcolumn: name of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: string
    :type systems: list (providing file name, information on tab with system output and system name for each element)
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations


# In[ ]:


def main(my_args=None):
    '''
    A main function. This does not make sense for a notebook, but it is here as an example.
    sys.argv is a very lightweight way of passing arguments from the commandline to a script.
    '''
    if my_args is None:
        my_args = sys.argv
    
    system_info = create_system_information(my_args[3:])
    evaluations = run_evaluations(my_args[1], my_args[2], system_info)
    provide_output_tables(evaluations)

if __name__ == '__main__':
    main()

# ## Evaluating spaCy and Stanford

# In[16]:


# my_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/spacy_out.dev-preprocessed.conll',2,'spacy','../data/stanford_out.dev-preprocessed.conll',3,'stanford']
# main(my_evaluations)


# ## Evaluating my systems

# ### Logreg

# In[18]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out.logreg.conll',4,'my logreg']
# main(mysystem_evaluations)


# ### NB

# In[19]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out.NB.conll',4,'my NB']
# main(mysystem_evaluations)


# ### SVM

# In[20]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out.SVM.conll',4,'my SVM']
# main(mysystem_evaluations)


# ## Evaluating my systems with features

# ### Logreg w/features

# In[29]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out_with_features.logreg.conll',4,'my logreg w/features']
# main(mysystem_evaluations)


# ### NB w/features

# In[28]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out_with_features.NB.conll',4,'my NB w/features']
# main(mysystem_evaluations)


# ### SVM w/features

# In[26]:


# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out_with_features.SVM.conll',4,'my SVM w/features']
# main(mysystem_evaluations)


# ### SVM - embedding

# In[33]:


# Only token
# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out_embedding.SVM.conll',4,'my SVM embedding']
# main(mysystem_evaluations)


# In[32]:


# with sparse vectors
# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out_embedding.embedding.conll',4,'my SVM embedding']
# main(mysystem_evaluations)


# In[ ]:

# CRF
# mysystem_evaluations = [0, '../data/conll2003.dev-preprocessed.conll',3,'../data/my_out.CRF.conll',1,'my CRF']
# main(mysystem_evaluations)
