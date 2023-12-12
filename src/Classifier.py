#----- Hierarhical classifier models -----#

# Imports
from Evaluation import evaluate_model
from AttentMechanisms import label_wise_attention, output_layer
from Utility import get_dataset, get_data_for_node

import pandas as pd
import numpy as np
import re
import time
import logging
import gc

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# system = 'mac'
system = 'ubuntu'

if system == 'mac':
    # MAC
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Input, Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Flatten, concatenate, Layer, SimpleRNN, LSTM, Bidirectional
    import tensorflow.keras.backend as K
else:
    # UBUNTU
    from keras_preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from keras.models import Model, Sequential
    from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Input, Flatten, concatenate, Layer, SimpleRNN, LSTM, Bidirectional
    from keras import Model
    import keras.backend as K

# Environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
pd.options.mode.chained_assignment = None
# tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Global variables
maxlen = 512
test_size = 0.15
level_2_thresholds = [140, 200, 170, 110, 160, 95, 145, 130, 140, 95, 130]

# Driver for the different model types
def main():

    # bert_words_cnn_model('single')
    # bert_words_cnn_model('multi')
    # bert_words_cnn_model('labelwise_attention')
    # bert_words_cnn_model('embedding_attention')
    bert_words_cnn_model('rnn')
    
    # lda_words_cnn_model('single')
    # lda_words_cnn_model('multi')
    # lda_words_cnn_model('attention')

    # bert_lda_words_cnn_model('single')
    # bert_lda_words_cnn_model('multi')
    # bert_lda_words_cnn_model('attention')
   
    # bert_lda_words_docs_cnn_model('single')
    # bert_lda_words_docs_cnn_model('multi')
    # bert_lda_words_docs_cnn_model('attention')

# Classifies papers by using BERT word embeddings as input to CNN
def bert_words_cnn_model(setting):

    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    classify(X_train, X_test, setting)

# Classifies papers by using LDA word embeddings as input to CNN
def lda_words_cnn_model(setting):

    X_train = np.load("data/ldaWordsTrainBig.npy", mmap_mode="r")
    X_test = np.load("data/ldaWordsTestBig.npy", mmap_mode="r")
    classify(X_train, X_test, setting)

# Classifies papers by using BERT and LDA word embeddings as input to CNN
def bert_lda_words_cnn_model(setting):

    X_train = np.load("data/bertAndLDAWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertAndLDAWordsTest.npy", mmap_mode="r")
    classify(X_train, X_test, setting)

# Classifies papers by using BERT and LDA document vectors as input to CNN
def bert_lda_documentvecs_cnn_model(setting):

    CLS_train = np.load("data/CLStrain.npy")
    CLS_test = np.load("data/CLStest.npy")

    lda_docs_train = np.load("data/ldaDocumentsTrainBig.npy")
    lda_docs_test = np.load("data/ldaDocumentsTestBig.npy")

    X_train = np.zeros((len(CLS_train), 2, 768))
    for i in range(0, len(X_train)):
            X_train[i, 0] = CLS_train[i]
            X_train[i, 1] = lda_docs_train[i]

    X_test = np.zeros((len(CLS_test), 2, 768))
    for i in range(0, len(X_test)):
            X_test[i, 0] = CLS_test[i]
            X_test[i, 1] = lda_docs_test[i]

    classify(X_train, X_test, setting)

# Classifies papers by using BERT and LDA word embeddings and document vectors as input to CNN
def bert_lda_words_docs_cnn_model(setting):

    X_train = np.load("data/allTogetherTrain.npy", mmap_mode="r")
    X_test = np.load("data/allTogetherTest.npy", mmap_mode="r")
    classify(X_train, X_test, setting)

# Used to classify research outputs in hierarchical structure
def classify(X_train, X_test, setting):
    placeholder, data = get_dataset()

    if setting in ['labelwise_attention', 'multi_labelwise_attention']:
        level_1_classifier = label_wise_attention_cnn_model
        level_2_classifier = label_wise_attention_cnn_model
    elif setting in ['embedding_attention', 'multi_embedding_attention']:
        level_1_classifier = cnn_model
        level_2_classifier = label_embedding_attention_cnn_model
        query_matrix = np.load("data/bertWordsDescription.npy", mmap_mode="r")
        new_query_matrix = []
        new_query_matrix.append(query_matrix[0:2])
        new_query_matrix.append(query_matrix[2:4])
        new_query_matrix.append(query_matrix[4:6])
        new_query_matrix.append(query_matrix[6:9])
        new_query_matrix.append(query_matrix[9:12])
        new_query_matrix.append(query_matrix[12:14])
        new_query_matrix.append(query_matrix[14:17])
        new_query_matrix.append(query_matrix[17:21])
        new_query_matrix.append(query_matrix[21:24])
        new_query_matrix.append(query_matrix[24:26])
        new_query_matrix.append(query_matrix[26:30])
    elif setting == 'rnn':
        level_1_classifier = rnn_model
        level_2_classifier = rnn_model
    else:
        level_1_classifier = cnn_model
        level_2_classifier = cnn_model
       
    temp_file = "data/temp_file"

    # Build root classifier for level 1 categories    
    level_1_data = data.loc[:, '1':'20']
    level_1_y = level_1_data.to_numpy()
    level_1_num_labels = level_1_y.shape[1]
    y_train, y_test = train_test_split(level_1_y, test_size = test_size, random_state = 1000)
    level_1_X, level_1_y = get_data_for_node(X_train, y_train, 500)
   
    if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")
    np.save(temp_file, level_1_X)
    del level_1_X
    gc.collect()
    level_1_X = np.load(temp_file + ".npy", mmap_mode="r")
    root_classifier = level_1_classifier(level_1_X, level_1_y, setting)
    del level_1_X
    gc.collect()
    if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")

    # Build classifiers using data from subcategories in level 2
    level_2_start_labels = ['101', '401', '702', '806', '907', '1101', '1302', '1404', '1702', '1808', '2003']
    level_2_end_labels = ['199', '499', '799', '899', '999', '1199', '1399', '1499', '1799', '1899', '2099']

    category_lengths = []
    level_1_classifiers = []
    for i in range(0, len(level_2_start_labels)):
        level_2_data = data.loc[:, level_2_start_labels[i]:level_2_end_labels[i]]
        level_2_labels = level_2_data.to_numpy()
        category_lengths.append(level_2_labels.shape[1])
        y_train, y_test = train_test_split(level_2_labels, test_size = test_size, random_state = 1000)
        temp_X_train, temp_y_train = get_data_for_node(X_train, y_train, level_2_thresholds[i])
        if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")
        np.save(temp_file, temp_X_train)
        del temp_X_train
        gc.collect()
        temp_X_train = np.load(temp_file + ".npy", mmap_mode="r")
        if setting in ['embedding_attention', 'multi_embedding_attention']:
            level_1_classifiers.append(level_2_classifier(temp_X_train, temp_y_train, new_query_matrix[i], setting))
        else:
            level_1_classifiers.append(level_2_classifier(temp_X_train, temp_y_train, setting))
        del temp_X_train
        gc.collect()
    if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")
    level_2_num_labels = np.sum(category_lengths)
    level_2_category_indices = [0] * len(category_lengths)
    for i in range(1, len(level_2_category_indices)):
        level_2_category_indices[i] = level_2_category_indices[i - 1] + category_lengths[i - 1]

    # Predict level 1   
    if setting in ['single', 'labelwise_attention', 'embedding_attention', 'rnn']:
        level_1_y_pred = root_classifier.predict([X_test])
    elif setting in ['multi', 'multi_labelwise_attention', 'multi_embedding_attention']:
        level_1_y_pred = root_classifier.predict([X_test, X_test, X_test])

    level_2_y_pred = []
    level_2_labels = data.columns.values[level_1_num_labels:]
    level_2_other_labels_indices = [i for i, label in enumerate(level_2_labels) if '99' in label]

    level_2_confidence_threshold = 0.75

    final_predictions = []
    for i in range(0,  X_test.shape[0]):
        # Get the predicted level 1 output
        level_1_max_index = np.where(level_1_y_pred[i] == np.amax(level_1_y_pred[i]))[0][0]
        level_1_y_pred[i] = [0] * len(level_1_y_pred[i])
        level_1_y_pred[i][level_1_max_index] = 1
        level_1_classifier = level_1_classifiers[level_1_max_index]
        if setting in ['single', 'labelwise_attention', 'embedding_attention', 'rnn']:
            temp_level_2_y_pred = np.squeeze(level_1_classifier.predict([X_test[i:i+1]]))
        elif setting in ['multi', 'multi_labelwise_attention', 'multi_embedding_attention']:
            temp_level_2_y_pred = np.squeeze(level_1_classifier.predict([X_test[i:i+1], X_test[i:i+1], X_test[i:i+1]]))

        # Get the predicted level 2 output by only looking at subcategory of level 1 output
        level_2_max_index = np.where(temp_level_2_y_pred == np.amax(temp_level_2_y_pred))[0][0]
        level_2_confidence = temp_level_2_y_pred[level_2_max_index]
        level_2_y_pred.append([0] * level_2_num_labels)

        current_level_2_num_labels = len(temp_level_2_y_pred)
        # if (level_2_confidence >= base_level_2_confidence_threshold/current_level_2_num_labels):
        if (level_2_confidence >= level_2_confidence_threshold):
            level_2_y_pred[i][level_2_category_indices[level_1_max_index] + level_2_max_index] = 1
        for j in level_2_other_labels_indices:
            level_2_y_pred[i][j] = 0
        final_predictions.append([*level_1_y_pred[i], *level_2_y_pred[i]])
    
    final_predictions = np.array(final_predictions)

    all_labels = data.to_numpy()
    y_train, y_test = train_test_split(all_labels, test_size = test_size, random_state = 1000)

    level_2_other_labels_indices = [x+level_1_num_labels for x in level_2_other_labels_indices]

    level_1_y_test = []
    for i in range(y_test.shape[0]):
        for j in level_2_other_labels_indices:
            y_test[i][j] = 0
        level_1_y_test.append(y_test[i][0:level_1_num_labels])

    level_2_y_test = []
    for i in range(y_test.shape[0]):
        level_2_y_test.append(y_test[i][level_1_num_labels:])
    
    level_1_y_test = np.array(level_1_y_test)
    level_1_y_pred = np.array(level_1_y_pred)
    level_2_y_test = np.array(level_2_y_test)
    level_2_y_pred = np.array(level_2_y_pred)

    if os.path.exists("results.txt"):
            os.remove("results.txt")
    f = open("results.txt", "a")
    print("Level 1 RESULTS:", file = f)
    evaluate_model(level_1_y_pred, level_1_y_test, f, 1)
    print("Level 2 RESULTS:", file = f)
    evaluate_model(level_2_y_pred, level_2_y_test, f, 2)
    print("Combined RESULTS:", file = f)
    evaluate_model(final_predictions, y_test, f, 3)
    f.close()

# CNN model used to classify research outputs
def cnn_model(X_train, y_train, setting):
    shape = X_train.shape[1:]

    epochs = 10
    batch_size = 32

    if 'multi' in setting:
        kernel_size_1 = 2
    else:
        kernel_size_1 = 3

    kernel_size_2 = 4
    kernel_size_3 = 6
    filter_size = 100

    #Build model
    inputs1 = Input(shape=shape)
    conv1 = Conv1D(filters=filter_size, kernel_size = kernel_size_1, activation='relu')(inputs1)
    pool1 = GlobalMaxPool1D()(conv1)
    dropout1 = Dropout(0.2)(pool1)
    flat1 = Flatten()(dropout1)

    inputs2 = Input(shape=shape)
    conv2 = Conv1D(filters=filter_size, kernel_size = kernel_size_2, activation='relu')(inputs2)
    pool2 = GlobalMaxPool1D()(conv2)
    flat2 = Flatten()(pool2)

    inputs3 = Input(shape=shape)
    conv3 = Conv1D(filters=filter_size, kernel_size = kernel_size_3, activation='relu')(inputs3)
    pool3 = GlobalMaxPool1D()(conv3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])
   
    if setting in ['single', 'labelwise_attention', 'embedding_attention']:
        outputs = Dense(y_train.shape[1], activation='softmax')(flat1)
        model = Model(inputs=[inputs1], outputs=outputs)
    elif setting in ['single', 'multi_labelwise_attention', 'multi_embedding_attention']:
        outputs = Dense(y_train.shape[1], activation='softmax')(merged)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    print(model.summary())

    #Train model with data
    if setting in ['single', 'labelwise_attention', 'embedding_attention']:
        model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2)
    elif setting in ['single', 'multi_labelwise_attention', 'multi_embedding_attention']:
        model.fit([X_train, X_train, X_train], y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2)
    return model

def rnn_model(X_train, y_train, setting):
    shape = X_train.shape[1:]
    epochs = 3
    batch_size = 128
    lstm_units = 128

    #Build model
    inputs = Input(shape=shape)
    lstm = LSTM(units=lstm_units, activation='tanh')(inputs)
    outputs = Dense(y_train.shape[1], activation='softmax')(lstm)
    model = Model(inputs=[inputs], outputs=outputs)

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    #Train model with data
    model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size)
    return model

#----- Attention mechanisms -----#
def label_wise_attention_cnn_model(X_train, y_train, setting):
    shape = X_train.shape[1:]
    epochs = 10
    batch_size = 32

    kernel_size_1 = 2
    kernel_size_2 = 4
    kernel_size_3 = 6
    filter_size = 100

    num_classes = y_train.shape[1]

    # Build model
    inputs1 = Input(shape=shape)
    conv1 = Conv1D(filters=filter_size, kernel_size = kernel_size_1, activation='relu')(inputs1)
    attention_layer1 = label_wise_attention(num_classes)(conv1)

    inputs2 = Input(shape=shape)
    conv2 = Conv1D(filters = filter_size, kernel_size = kernel_size_2, activation='relu')(inputs2)
    attention_layer2 = label_wise_attention(num_classes)(conv2)

    inputs3 = Input(shape=shape)
    conv3 = Conv1D(filters = filter_size, kernel_size = kernel_size_3, activation='relu')(inputs3)
    attention_layer3 = label_wise_attention(num_classes)(conv3)

    if setting == 'labelwise_attention':
        outputs = output_layer(num_classes)(attention_layer1)
        model = Model(inputs=[inputs1], outputs=outputs)
    elif setting == 'multi_labelwise_attention':
        attention_layers = concatenate([attention_layer1, attention_layer2,attention_layer3])
        outputs = output_layer(num_classes)(attention_layers)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    print(model.summary())

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    #Train model with data
    if setting == 'labelwise_attention':
        model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size)
    elif setting == 'multi_labelwise_attention':
        model.fit([X_train, X_train, X_train], y_train, epochs = epochs, batch_size = batch_size)

    return model

def label_embedding_attention_cnn_model(X_train, y_train, query_matrix, setting):
    shape = X_train.shape[1:]
    epochs = 5
    batch_size = 32

    kernel_size_1 = 2
    kernel_size_2 = 4
    kernel_size_3 = 6
    kernel_size_4 = 8
    filter_size = 100

    num_classes = y_train.shape[1]

    # Build model
    inputs1 = Input(shape=shape)
    conv1 = Conv1D(filters=filter_size, kernel_size = kernel_size_1, activation='relu')(inputs1)
    attention_layer1 = general_attention()([conv1, query_matrix])

    if setting == 'embedding_attention':
        outputs = output_layer(num_classes)(attention_layer1)
        model = Model(inputs=[inputs1], outputs=outputs)
    elif setting == 'multi_embedding_attention':
        inputs2 = Input(shape=shape)
        conv2 = Conv1D(filters=filter_size, kernel_size = kernel_size_2, activation='relu')(inputs2)
        attention_layer2 = general_attention()([conv2, query_matrix])

        inputs3 = Input(shape=shape)
        conv3 = Conv1D(filters=filter_size, kernel_size = kernel_size_3, activation='relu')(inputs3)
        attention_layer3 = general_attention()([conv3, query_matrix])

        attention_layers = concatenate([attention_layer1, attention_layer2, attention_layer3])
        outputs = output_layer(num_classes)(attention_layers)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    print(model.summary())

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    #Train model with data
    if setting == 'embedding_attention':
        model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size)
    elif setting == 'multi_embedding_attention':
        model.fit([X_train, X_train, X_train], y_train, epochs = epochs, batch_size = batch_size)

    return model

if __name__ == '__main__':
    main()