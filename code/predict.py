from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    import tensorflow.keras as K
    import numpy as np
    import sys
    sys.path.insert(0, resources_path)          # enables to import files
    from create_vocabularies import getChars, getBigrams, collect_vocabularies, char2array, bigram2array
    
    MAX_LENGTH = 50

    file_in = open(input_path, encoding='utf-8')            # opens the file to predict
    file_out = open(output_path, 'w', encoding='utf-8')     # opens the file with the predictions

    file_temp_out = open('./temp.UTF8', 'w', encoding='utf-8')   # a temporary file of support
    
    lengths = []            # list of all the true lengths of each sentence of the input file
    subdivision = []        # list that contains the number of subdivision for each sentence, in order to reconstruct them after for providing a label for each character
    for line in file_in:
        x_len = len(line.strip())
        ratio = int(np.ceil(x_len/MAX_LENGTH))
        subdivision.append(ratio)       # updates the subdivision list in the amount of lines the original one will be divided
        for times in range(ratio-1):
            lengths.append(MAX_LENGTH)    # updates the length with the maximum one (because the sentence is more than MAX_LENGTH)
            file_temp_out.write(line[times*MAX_LENGTH:((times+1)*MAX_LENGTH)]+'\n') # writes in the file the sentance truncated
        h = MAX_LENGTH*(ratio-1)
        lengths.append(x_len-h)    # updates the length with the remaining length of the sentence
        file_temp_out.write(line[h:])  # writes in the file the remaining sentence
    file_temp_out.close()

    char2id, bigram2id, id2char, id2bigram = collect_vocabularies(resources_path+'/char_model_train', resources_path+'/bigram_model_train')
    char_input = char2array('./temp.UTF8', char2id, MAX_LENGTH)    # create the numpy array to predict from the splitted version for char and bigrams
    bigram_input = bigram2array('./temp.UTF8', bigram2id, MAX_LENGTH)
    
    model = K.models.load_model(resources_path+'/modelBest2.h5')    # load the model

    predictions = model.predict(x=[char_input, bigram_input])   # predict the values
    predictions = np.argmax(predictions, axis=2)    # take from the predictions the index corresponding to the highest probability
    dictionary = {0:'B', 1:'I', 2:'E', 3:'S'}
    file_temp_out = open('./temp.UTF8', 'w', encoding='utf-8')
    for i in range(predictions.shape[0]):   # for each sentence
        outs = ''                       # will contain the string of labels
        for j in range(lengths[i]):    # until the true length of that sentence     
            outs = outs + dictionary[predictions[i,j]] # updates with the corresponding label      
        file_temp_out.write(outs+'\n')  # updates the temporary file with the subdivided labels
    file_temp_out.close()

    file_temp_out = open('./temp.UTF8', encoding='utf-8')
    for elem in subdivision:        # reconstructs the labels looking at how many times a sentence was subdivided
        labels = ''
        for times in range(elem):
            labels += file_temp_out.readline().strip()
        file_out.write(labels+'\n') # updates the final file with one sentence of labels for each original sentence
    file_temp_out.close()
    file_in.close()
    file_out.close()
    pass


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
