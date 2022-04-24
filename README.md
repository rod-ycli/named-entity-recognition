# ma-ml4nlp-labs

## overview

This repository provides scripts for the final NER portfolio of student YC Roderick Li (2740992) for the course 'Machine Learning in NLP'.

This repository needs to be placed together with a "data" and a "model" directory to operate.

The scripts correlate to different sections of the portfolio. They are meant to be executed in the following sequence:

1. Preprocessing_conll.py -> Experimental setup

    The script strips the B- and I- label off the conll.dev set and the conll.train set. It also align the spaCy and Stanford entries and convert their prediction labels to the gold labels.
    It takes no argument in the command line. By default, it needs the following filepaths to run:
    1. '../data/spacy_out.dev.conll'
    2. '../data/stanford_out.dev.conll
    3. '../data/conll2003.dev.conll'
    4. '../data/conll2003.train.conll'
    Executing this file will result in the following ouputs:
    1. Preprocess all the above file and save as '../data/[basename]-preprocessed.conll'
    2. Write out a conversion.tsv in the /settings directory.

2. Systems.py -> Results of basic systems

    This script creates 3 basic NER systems (namely Logreg, NB and SVM) taking only the tokens as a feature, 3 with more features, an SVM taking word embedding as features and an SVM mixing dense and sparse vectors. There will be 8 prediction outputs created in "../data/" in total, along with 2 "Features" files with features extracted from each entry.
    It takes three arguments. User can define them, or edit the script to use the default arguments:
    1. trainingfile: the filepath of the training data (default = "../data/conll2003.train-preprocessed.conll)
    2. inputfile: the filepath of the validation/test data (default = "../data/conll2003.dev-preprocessed.conll")
    3. outputfile: the filepath of the prediction output (default = "../data/my_out.conll")

3. Results_and_basic_evaluation.py -> Results Evaluation

    This script returns the confusion matrix and the evaluation scores of a particular prediction output. It is used to present the results of the systems.
    It can take several arguments:
    1. validation/test set filepath = '../data/conll2003.dev-preprocessed.conll'
    2. column of gold label = 3
    Then the following three arguments indicate system info:
        (1) prediction filepath
        (2) prediction column
        (3) system name
    More than one set of system info can be passed.

4. Features_ablation_analysis.py -> Feature ablation analysis

    This script automatically runs feature ablation analyses on the 3 systems with each additional feature paired with the token, and prints the confusion matrix and evaluation table for each test.
    It will use the same trainingfile and inputfile documented above.

5. ErrorAnalysis.py -> Error Analysis

    This script looks into the difference with/without the features, and also map the changes in the results to a certain feature observed in the training data.
    By default, it investigates the effect of "POS" on the PER label, and the "next token" on ORG.
    It uses the train, dev and the prediction results conll files placed in "../data".

