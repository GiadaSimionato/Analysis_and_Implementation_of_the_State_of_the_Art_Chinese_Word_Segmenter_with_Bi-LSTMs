# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create_vocabularies script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains all the functions useful to create the vocabularies and to load and convert the data in order to feed the network.
#
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

from gensim.models import Word2Vec
import numpy as np


# -- Function that returns a list of characters from a line --
# --- :param line: line to trim.
# --- :return list: list of the characters that compose the line. 

def getChars(line):
    return list(line)


# -- Function that returns a list of bigrams from a line --
# --- :param line: line to trim.
# --- :return list: list of the bigrams that compose the line.

def getBigrams(line):
    bigrams = []
    for i in range(len(line)-1):
        bigrams.append(line[i]+line[i+1])
    return bigrams


# -- Function that creates and saves the models of the vocabularies --
# --- :param path: the path of the file from which create the models of the vocabularies.
# --- :param char_emb_size: the size of the embedding for characters.
# --- :param bigram_emb_size: the size of the embedding for bigrams.
# --- :param to_train: flag that encodes the possibility to train the embeddings in case these Gensim models are used in place of the Keras 'Embedding' layer.
# --- :param train_epochs: number of epochs for the training of the models.
# --- :return None: saves the models in the 'resources' folder.

def create_models(path, char_emb_size, bigram_emb_size, to_train=False, train_epochs=5):

    f = open(path, encoding='utf-8')
    char_doc = []
    bigram_doc = []
    line = f.readline()
    while line != '':                       # until the document is not over
        line = line.strip()                 # removes the '\n' at the end of the line
        charList = getChars(line)           # obtains the character list from the line
        bigramList = getBigrams(line)       # obtains the bigram list from the line
        char_doc.append(charList)           # updates the list of lists of the characters
        bigram_doc.append(bigramList)       # updates the list of lists of the bigrams
        line = f.readline()

    char_model = Word2Vec(char_doc, size=char_emb_size)         # creates the character model with the list of list of characters
    bigram_model = Word2Vec(bigram_doc, size=bigram_emb_size)   # creates the bigram model with the list of list of bigrams
    if to_train:                                                # if it's required too train them, then it trains for train_epochs times
        char_model.train(char_doc, total_examples=len(char_doc), epochs=train_epochs)
        bigram_model.train(bigram_doc, total_examples=len(bigram_doc), epochs=train_epochs)

    char_model.save('../resources/char_model_train')        # saves the character model in the resources folder
    bigram_model.save('../resources/bigram_model_train')    # saves the bigram model in the resources folder

    f.close()
    return


# -- Function that extracts the four vocabularies from the char and bigrams models --
# --- :param path_char_model: the path to the Gensim char model.
# --- :param path_bigram_model: the path to the Gensim bigram model.
# --- :return char2id: a dictionary with as keys all the characters provided by the char model and as values their index, augmented with the UNK char and the PAD one.
# --- :return bigram2id: a dictionary with as keys all the bigrams provided by the bigram model and as values their index, augmented with the UNK char and the PAD one.
# --- :return id2char: a dictionary with as keys all the indices and as values the corresponding characters provided by the char model, augmented with the UNK char and the PAD one.
# --- :return id2bigram: a dictionary with as keys all the indices and as values the corresponding bigrams provided by the bigram model, augmented with the UNK char and the PAD one.

def collect_vocabularies(path_char_model, path_bigram_model):

    char_model = Word2Vec.load(path_char_model)         # loads the char model
    bigram_model = Word2Vec.load(path_bigram_model)     # loads the bigram model

    char_list = char_model.wv.index2word        # saves the KeyedVector object of the model that contains the list of all the characters with a frequency of 3 or more occurencies
    char2id = dict()
    char2id['<PAD>'] = 0    # augments the dictionary with the PAD element
    char2id['<UNK>'] = 1    # augments the dictionary with the UNK element
    char2id.update({v:k+len(char2id) for k, v in enumerate(char_list)})     # converts to dictionary the KeyedVector's list.

    bigram_list = bigram_model.wv.index2word    # does the same for the bigram model
    bigram2id = dict()
    bigram2id['<PAD>'] = 0
    bigram2id['<UNK>'] = 1
    bigram2id.update({v:k+len(bigram2id) for k, v in enumerate(bigram_list)})

    id2char = {v:k for k,v in char2id.items()}          # creates the reversed vocabulary: as keys the indices and as values the corresponding characters
    id2bigram = {v:k for k,v in bigram2id.items()}      # creates the reversed vocabulary: as keys the indices and as values the corresponding bigrams

    return char2id, bigram2id, id2char, id2bigram


# -- Function that converts a file in a numpy array of indices at character-level  --
# --- :param path: the path of the file to convert.
# --- :param voc: the vocabulary from character representation to index (char2id).
# --- :param maxLen: the maximum length of the sentences to convert.
# --- :return char_dataset: a numpy array of dimensions (length_of_document, maxLen) containing the indices of the first maxLen characters of each sentence in the file.

def char2array(path, voc, maxLen):

    char_dataset = []
    f = open(path, encoding='utf-8')            # opens the file
    for line in f:                              # until the document is not over
        line = line.strip()                     # removes the '\n' at the end of the line
        charList = getChars(line)               # obtains the list of characters in the line
        ind = np.zeros((1, maxLen), dtype=int)  # create an array of zeros of length maxLen (to optimize the padding step)
        if len(charList)>maxLen:                # truncates all the sentences longer than maxLen to maxLen (this does not affect the training step)
            charList = charList[:maxLen]
        for i, char in enumerate(charList):     # for each character in the list inserts in the corresponding position the index of the character if the char is in the
            if char in voc:                     # vocabulary or the index of 'UNK' otherwise.
                ind[0,i] = voc[char]
            else:
                ind[0,i] = voc['<UNK>']
        char_dataset.append(ind)                # append this array to the one that collects all the arrays for each line
    f.close()
    return np.squeeze(np.asarray(char_dataset)) # return the numpy format for this array (instead of the list)


# -- Function that converts a file in a numpy array of indices at bigram-level  --
# --- :param path: the path of the file to convert.
# --- :param voc: the vocabulary from bigram representation to index (bigram2id).
# --- :param maxLen: the maximum length of the sentences to convert .
# --- :return bigram_dataset: a numpy array of dimensions (length_of_document, maxLen) containing the indices of the first maxLen bigrams of each sentence in the file.

def bigram2array(path, voc, maxLen):      # works as char2array
    
    bigram_dataset = []
    f = open(path, encoding='utf-8')
    for line in f:
        line = line.strip()
        bigramList = getBigrams(line)
        ind = np.zeros((1,maxLen))
        if len(bigramList)>maxLen:
            bigramList = bigramList[:maxLen]
        for i, bigram in enumerate(bigramList):
            if bigram in voc:
                ind[0,i] = voc[bigram]
            else:
                ind[0,i] = voc['<UNK>']
        bigram_dataset.append(ind)
    f.close()
    return np.squeeze(np.asarray(bigram_dataset))


# -- Function that converts a label from BIES format to the corresponding vector  --
# --- :param label: the label 'B', 'I' , 'E' or 'S' to convert.
# --- :return vector: the 4-dims vector corresponding to label.

def convert_label(label):

    dictionary = {'B': np.array([1,0,0,0]), 'I': np.array([0,1,0,0]), 'E': np.array([0,0,1,0]), 'S': np.array([0,0,0,1])}
    return dictionary[label]


# -- Function that creates a numpy array that contains all the vectors corresponding to the labels in the BIES format from a file --
# --- :param path: the path of the file in BIES format to convert.
# --- :param max_length: the maximum length of the sentences to convert.
# --- :return labVect: a numpy array with dimensions (lenght_of_document, max_length, 4) containing the vector format of the first max_length labels for each sentence in the file.

def labels2vect(path, max_length):

    f = open(path, encoding='utf-8')
    labVect = []
    for line in f:                                      # for each line in the file
        line = line.strip()
        labList = getChars(line)
        if len(labList)>max_length:                     # truncates the label sentence to max_length if longer
            labList = labList[:max_length] 
        vec = np.zeros((max_length,4), dtype=int)      
        for i, label in enumerate(labList):
            vec[i,:] = convert_label(label)             # updates the numpy array of zeros with the vector format of each label
        labVect.append(vec)
    f.close()
    return np.asarray(labVect)                          # return the total array in such format