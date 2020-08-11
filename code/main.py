# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains the main of the implementation.
# For the detailed explanations of the functions used in this script please refer to the corresponding files in the 'code' folder.
#
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from gensim.models import Word2Vec
from create_vocabularies import create_models, collect_vocabularies, char2array, bigram2array, labels2vect
from collect_network import build_model

# HYPERPARAMETERS INITIALIZATIONS

CHAR_EMBEDDING_SIZE = 64
BIGRAM_EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512		# number of hidden units for the first bidirectional LSTM layer
HIDDEN_SIZE2 = 256		# number of hidden units for the second and third bidirectional LSTM layers
DROPOUT = 0.2			# input dropout
REC_DROPOUT = 0.2		# recurrent dropout for the first layer
REC_DROPOUT2 = 0.33		# recurrent dropout for the second and third layers

BATCH_SIZE = 32
EPOCHS = 5

MAX_LENGTH = 50

# PATHS DEFINITIONS 

path_train = 'Dataset/training/as_training_simply.UTF8'					# path of the training file
path_test = 'Dataset/testing/as_test_simply.UTF8'						# path of the testing file
path_labels_train = 'Dataset/gold/as_training_simply_labels.UTF8'		# path of the gold truth for training
path_labels_test = 'Dataset/gold/as_test_simply_labels.UTF8'			# path of the gold truth for testing

path_char_model = '../resources/char_model_train'						# path of the char model
path_bigram_model = '../resources/bigram_model_train'					# path of the bigram model

create_models(path_train, CHAR_EMBEDDING_SIZE, BIGRAM_EMBEDDING_SIZE)	# creates and saves the char and bigram models 

char2id, bigram2id, id2char, id2bigram = collect_vocabularies(path_char_model, path_bigram_model)	# collects the four vocabularies

print('Creating training set...')
char_input_train = char2array(path_train, char2id, MAX_LENGTH)			# creates the array of input for the chars
bigram_input_train = bigram2array(path_train, bigram2id, MAX_LENGTH)	# creates the array of input for the bigrams
train_labels = labels2vect(path_labels_train, MAX_LENGTH)				# creates the array of the labels
print('Done.')

print('Creating validation set...')
char_input_test = char2array(path_test, char2id, MAX_LENGTH)			# repeats the same process for the validation set
bigram_input_test = bigram2array(path_test, bigram2id, MAX_LENGTH)
test_labels = labels2vect(path_labels_test, MAX_LENGTH)
print('Done.')

char_vocab_size = len(char2id.keys())		# computes the length of the char vocabulary
bigram_vocab_size = len(bigram2id.keys())   # computes the length of the bigram vocabulary

model = build_model(char_vocab_size, CHAR_EMBEDDING_SIZE, bigram_vocab_size, 
					BIGRAM_EMBEDDING_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, DROPOUT,
					REC_DROPOUT, REC_DROPOUT2, MAX_LENGTH)		# creates the model of the neural network 

cbk = K.callbacks.TensorBoard("logging/best_model")		# initializes the callbacks for Tensorboard
print("Starting training...")

model.fit(x=[char_input_train, bigram_input_train], y=train_labels, batch_size=BATCH_SIZE, 
	       epochs=EPOCHS, verbose=1, validation_data=([char_input_test, bigram_input_test], test_labels), callbacks=[cbk]) # trains the model

model.save('modelBest.h5')		# saves the model

print("Training completed.")
