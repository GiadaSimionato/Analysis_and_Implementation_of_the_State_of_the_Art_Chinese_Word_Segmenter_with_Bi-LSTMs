# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preprocessing script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It sets the environment. The 'icwb2-data' folder must be put in the same folder as 'code' one.
# It creates a folder 'Dataset' which contains three folders:
# - 'Training' contains four files that are the unsegmented version (produced by this script) of the four datasets (two of which were simplified).
# - 'Testing' contains four files that are the copied test files (two of which were simplified).
# - 'Gold' contains eight files, four are the labeled versions of the training datasets and four are the translated in BIES format of the gold of the test sets. 
#
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import pathlib
import re
from hanziconv import HanziConv
import unicodedata

trainFolder = '../icwb2-data/training/'       # path for the training folder of 'icwb2-data'
testFolder = '../icwb2-data/testing/'         # path for the test folder of 'icwb2-data'
goldFolder = '../icwb2-data/gold/'            # path for the gold folder of 'icwb2-data'

pathlib.Path('Dataset').mkdir(parents=True, exist_ok=True)              # creation of the nested folders
pathlib.Path('Dataset/training').mkdir(parents=True, exist_ok=True)
pathlib.Path('Dataset/testing').mkdir(parents=True, exist_ok=True)
pathlib.Path('Dataset/gold').mkdir(parents=True, exist_ok=True)

trainFiles = ["as_training_simply", "cityu_training_simply", "msr_training", "pku_training"]    # list of files to use in the three folders
testFiles = ['as_test_simply','cityu_test_simply','msr_test','pku_test']   
goldTestFiles = ['as_testing_gold_simply', 'cityu_test_gold_simply', 'msr_test_gold', 'pku_test_gold'] 

toSimplify = ["as_training", "cityu_training", "as_test", "cityu_test", "as_testing_gold", "cityu_test_gold"]   # list of files that must be simplified


# -- Function that creates the BIES-format label for a line --
# --- :param line: string to translate.
# --- :return label: corresponding label in BIES format.  

def to_segment_label(line):

    label = ""
    line = line.strip()             # removes the '\n' at the end of the line
    pieces = line.split()           # splits the line wrt '\u3000' for as, 'single_space' for cityu and 'double_spaces' for msr and pku
    for word in pieces:
        length = len(word)
        if(length==1):              # if the word is a single character than is represented by 'S'
            label = label + 'S'
        else:                       # otherwise the word contains at least two characters encoded as 'B(I)*E'
            label = label + 'B'
            for i in range(length-2):
                label = label + 'I'
            label = label + 'E'
    return label+'\n'


# -- Function that unsegments (remove spaces) from a line --
# --- :params line: string to unsegment.
# --- :return line: corresponding unsegmented string.

def to_unsegment(line):
    
    return re.sub(' +|\u3000', '', line)    # substitutes every expression corresponding to the RegEx ' +|\u3000' with '' in the line.


# ---- Part that creates the simplified versions (translation to Simplified Chinese) of the files that required this. ----

for f in range(len(toSimplify)):
    if f<2:                                             # to check in which folder the file to be simplified must be searched
        folder = trainFolder
    elif f==2 or f==3:
        folder = testFolder
    else:
        folder = goldFolder
    src_path = folder + toSimplify[f] + ".UTF8"         # composes the path of the file to be simplified
    dst_path = folder + toSimplify[f] + "_simply.UTF8"  # composes the path where to put the simplified version
    src_file = open(src_path, encoding='utf-8')         # opens the source file
    dst_file = open(dst_path, 'a', encoding='utf-8')    # opens the destination file
    line = src_file.readline()
    while(line!=""):                                    # until the document is not over
        simplified = HanziConv.toSimplified(line)       # simplifies the line
        dst_file.write(simplified)                      # updates the simplified document
        line = src_file.readline()
    src_file.close()                                    # closes the source file
    dst_file.close()                                    # closes the destination file


# ---- Part that unsegments and normalizes to half-width the training datasets and creates the relative labels. ----

for dataset in trainFiles:   
    src_path = trainFolder + dataset + '.UTF8'              # source path of the training file to unsegment
    dst_path = 'Dataset/training/' + dataset + '.UTF8'      # destination path of the unsegmented version
    label_path = 'Dataset/gold/'+ dataset + '_labels.UTF8'  # path of the resulting label file
    src_file = open(src_path, encoding='utf-8')             # opens the source file
    dst_file = open(dst_path, 'a', encoding='utf-8')        # opens the destination file
    label_file = open(label_path, 'a', encoding='utf-8')    # opens the label file
    line = src_file.readline()
    while(line!=""):                                        # until the document is not over
        if line != '\n':
            line = unicodedata.normalize('NFKC', line)          # converts all the digits, punctuation and Latin letters to half-width
            label = to_segment_label(line)                      # obtains the label of the line
            label_file.write(label)                             # updates the label document with the label
            line = to_unsegment(line)                           # obtains the unsegmented version of the line
            dst_file.write(line)                                # updates the dst document with the unsegmented line
        line = src_file.readline()
    src_file.close()                                        # closes the source file
    dst_file.close()                                        # closes the destination file
    label_file.close()                                      # closes the label file


# ---- Part that normalizes to half-width the test sets and converts the gold truth in BIES format.

for dataset in range(len(testFiles)):
    src = testFolder + testFiles[dataset] + '.UTF8'                         # source path of the test file
    dst = 'Dataset/testing/' + testFiles[dataset]+ '.UTF8'                  # destination path of the copy of the test file
    src_file = open(src, encoding='utf-8')                                  # opens the test file
    dst_file = open(dst, 'a', encoding='utf-8')                             # opens the destination of the test file
    line = src_file.readline()
    while line!="":                                                         # until the document is not over
        if line != '\n':
            line = unicodedata.normalize('NFKC', line)                          # converts all the digits, punctuation and Latin letters to half-width
            line = to_unsegment(line)                                           # removes spaces eventually left in the test set
            dst_file.write(line)                                                # updates the dst document with the normalized line
        line = src_file.readline()
    src_file.close()                                                        # closes the source file
    dst_file.close()                                                        # closes the destination file
    src_gold_test = goldFolder + goldTestFiles[dataset] + '.UTF8'           # source path of the gold labels of the test
    dst_gold_test = 'Dataset/gold/' + testFiles[dataset] +'_labels.UTF8'    # destination path of the translated labels for the test
    src_file = open(src_gold_test, encoding='utf-8')                        # opens the source file
    dst_file = open(dst_gold_test, 'a', encoding='utf-8')                   # opens the destination file
    line = src_file.readline()
    while(line!=""):                                                        # until the document is not over
        if line != '\n':
            line = unicodedata.normalize('NFKC', line)                          # converts all the digits, punctuation and Latin letters to half-width
            label = to_segment_label(line)                                      # obtains the label from the line
            dst_file.write(label)                                               # updates the destination file with the label
        line = src_file.readline()
    src_file.close()                                                        # closes the source file
    dst_file.close()                                                        # closes the destination file


