# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Collect_network script.
#
# @author Giada Simionato <simionato.1822614@studenti.uniroma1.it>
#
# It contains the function that builds the neural network architecture.
#
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from keras.layers import Masking, Dense, Concatenate, Bidirectional, Embedding, TimeDistributed, Input, LSTM
from keras.models import Model
from keras.optimizers import Adadelta, SGD, Adam

# -- Function that builds the neural network architecture --
# --- :param char_vocab_size: size of the char vocabulary.
# --- :param char_embedding_size: size of the char embedding.
# --- :param bigram_vocab_size: size of the bigram vocabulary.
# --- :param bigram_embedding_size: size of the bigram embedding.
# --- :param hidden_size: number of hidden units for the first layer.
# --- :param hidden_size2: number of hidden units for the second and third layers.
# --- :param p_dropout: probability of dropout for input.
# --- :param rec_drop: recurrent dropout for the first layer.
# --- :param rec_drop2: recurrent dropout for the second and third layers.
# --- :param max_len: maximum length.
# --- :return network: the model created. 




def build_model(char_vocab_size, char_embedding_size, bigram_vocab_size, bigram_embedding_size, hidden_size, hidden_size2, p_dropout, rec_drop, rec_drop2, max_len):

    input_chars = Input(shape=(max_len,))                # layer of input for chars
    input_bigrams = Input(shape=(max_len,))              # layer of input for bigrams
    char_mask = Masking(mask_value=0.0)(input_chars)     # mask layer for masking the padding in the char input layer
    bigram_mask = Masking(mask_value=0.0)(input_bigrams) # mask layer for masking the padding in the bigram input layer

    char_embeddings = Embedding(char_vocab_size, char_embedding_size, mask_zero=True)(char_mask)            # layer of embeddings for characters
    bigram_embeddings = Embedding(bigram_vocab_size, bigram_embedding_size, mask_zero=True)(bigram_mask)    # laver of embeddings for bigrams
    embeddings = Concatenate(axis=2)([char_embeddings, bigram_embeddings])                                  # concatenation of the char embeddings with bigram embeddings

    biLSTM1 = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=p_dropout, recurrent_dropout=rec_drop))(embeddings)  # first Bi-LSTM layer
    biLSTM2 = Bidirectional(LSTM(hidden_size2, return_sequences=True, dropout=p_dropout, recurrent_dropout=rec_drop))(biLSTM1)    # second Bi-LSTM layer
    biLSTM3 = Bidirectional(LSTM(hidden_size2, return_sequences=True, dropout=p_dropout, recurrent_dropout=rec_drop))(biLSTM2)    # third Bi-LSTM layer
    ffLayer = TimeDistributed(Dense(4, activation='softmax'))(biLSTM3)   # dense layer of 4 units

    network = Model(inputs=[input_chars, input_bigrams], outputs=ffLayer)   # defines the input and output layers

    network.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])    # set the model loss, optimizer and metrics for validation

    network.summary()   # summary of the architecture of the model

    return network