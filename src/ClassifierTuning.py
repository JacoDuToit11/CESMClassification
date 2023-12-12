#---- Imports -----#

# import functions from other files
from category_descriptions import get_level_2_descriptions
from Evaluation import save_results
from AttentMechanisms import label_wise_attention, output_layer
from Utility import get_dataset, get_data_for_node, predict, get_y_test

from npy_append_array import NpyAppendArray
import pandas as pd
import numpy as np
import logging
import gc
import time
from copy import deepcopy

from sklearn.model_selection import train_test_split
import keras_tuner as kt

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
    from keras.models import Model, Sequential, clone_model, load_model
    from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Input, Flatten, concatenate, Layer, SimpleRNN, LSTM, Bidirectional
    from keras import Model
    import keras.backend as K

#----- Environment variables -----#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
pd.options.mode.chained_assignment = None
# tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#----- Global variables -----#
maxlen = 512

#Extract all rows or just a certain number
allrows = True
num_rows = 2000

test_size = 0.15
level_2_thresholds = [89, 200, 170, 110, 160, 95, 145, 130, 73, 70, 130]

num_runs = 10

#----- Driver for the different model types -----#
def main():

    global file_name

    setting1 = 'single'
    setting2 = 'multi'
    setting3 = 'labelwise_attention'
    setting5 = 'rnn'

    settings = [setting1, setting2, setting3, setting4]

    for i in range(len(settings)):
        setting = settings[i]
        file_name = "Results/S1/" + setting + "_results.txt"
        bert_words_cnn_model_tuning(setting)

    for i in range(len(settings)):
        setting = settings[i]
        file_name = "Results/S2/" + setting + "_results.txt"
        bert_lda_words_cnn_model_tuning(setting)

    for i in range(len(settings)):
        setting = settings[i]
        file_name = "Results/S3/" + setting + "_results.txt"
        bert_lda_words_docs_cnn_model_tuning(setting)


# Classifies papers by using BERT word embeddings as input to CNN
def bert_words_cnn_model_tuning(setting):

    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    tune_and_classify(X_train, X_test, setting)

# Classifies papers by using BERT and LDA word embeddings as input to CNN
def bert_lda_words_cnn_model_tuning(setting):

    X_train = np.load("data/bertAndLDAWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertAndLDAWordsTest.npy", mmap_mode="r")
    tune_and_classify(X_train, X_test, setting)

# Classifies papers by using BERT and LDA word embeddings and document vectors as input to CNN
def bert_lda_words_docs_cnn_model_tuning(setting):

    X_train = np.load("data/allTogetherTrain.npy", mmap_mode="r")
    X_test = np.load("data/allTogetherTest.npy", mmap_mode="r")
    tune_and_classify(X_train, X_test, setting)

def tune_and_classify(X_train, X_test, setting):
    placeholder, data = get_dataset()

    if setting in ['labelwise_attention', 'multi_labelwise_attention']:
        level_1_hyper_model = LabelwiseAttentionCNNHyperModel
        level_2_hyper_model = LabelwiseAttentionCNNHyperModel
    elif setting == 'rnn':
        level_1_hyper_model = RNNHyperModel
        level_2_hyper_model = RNNHyperModel
    else:
        level_1_hyper_model = CNNHyperModel
        level_2_hyper_model = CNNHyperModel
       
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
    
    input_shape = level_1_X.shape[1:]
    num_classes = level_1_y.shape[1]

    max_epochs = 20
    factor = 3
    epochs = 20

    tuner = kt.Hyperband(level_1_hyper_model(input_shape, num_classes, setting), 
                        objective = 'val_loss', 
                        max_epochs = max_epochs, 
                        factor = factor, 
                        project_name = 'keras_tuner_file',
                        overwrite = True)
    
    level_1_classifier, best_epoch = tune_model(level_1_X, level_1_y, tuner, epochs, setting)

    # Train root classifiers for different runs
    base_level_1_classifier = clone_model(level_1_classifier)
    # Train root classifiers for different runs
    level_1_classifier_runs_weights = []
    for i in range(num_runs):
        if not 'multi' in setting:
            level_1_classifier.fit(level_1_X, level_1_y, epochs = best_epoch, validation_split = 0.2)
        else:
            level_1_classifier.fit([level_1_X, level_1_X, level_1_X], level_1_y, epochs = best_epoch, validation_split = 0.2)
        weights = level_1_classifier.get_weights()
        level_1_classifier_runs_weights.append(deepcopy(weights))

    # Build classifiers using data from subcategories in level 2
    level_2_start_labels = ['101', '401', '702', '806', '907', '1101', '1302', '1404', '1702', '1808', '2003']
    level_2_end_labels = ['199', '499', '799', '899', '999', '1199', '1399', '1499', '1799', '1899', '2099']
    num_level_2_classifiers = len(level_2_start_labels)

    max_epochs = 30
    factor = 3
    epochs = 30

    category_lengths = []
    level_2_classifiers_runs_weights = []
    level_2_classifiers = []
    for i in range(num_level_2_classifiers):
        print('CLASSIFIER ', i)
        level_2_data = data.loc[:, level_2_start_labels[i]:level_2_end_labels[i]]

        level_2_labels = level_2_data.to_numpy()
        category_lengths.append(level_2_labels.shape[1])
        y_train, y_test = train_test_split(level_2_labels, test_size = test_size, random_state = 1000)
        temp_X_train, temp_y_train = get_data_for_node(X_train, y_train, level_2_thresholds[i])
        temp_num_classes = temp_y_train.shape[1]

        if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")
        np.save(temp_file, temp_X_train)
        del temp_X_train
        gc.collect()
        temp_X_train = np.load(temp_file + ".npy", mmap_mode="r")

        tuner = kt.Hyperband(level_2_hyper_model(input_shape, temp_num_classes, setting), 
                            objective = 'val_loss', 
                            max_epochs = max_epochs, 
                            factor = factor, 
                            project_name = 'keras_tuner_file',
                            overwrite = True)

        temp_classifier, best_epoch = tune_model(temp_X_train, temp_y_train, tuner, epochs, setting)

        level_2_classifiers.append(clone_model(temp_classifier))

        temp_classifier_runs_weights = []
        for i in range(num_runs):
            if not 'multi' in setting:
                temp_classifier.fit(temp_X_train, temp_y_train, epochs = best_epoch, validation_split = 0.2)
            else:
                temp_classifier.fit([temp_X_train, temp_X_train, temp_X_train], temp_y_train, epochs = best_epoch, validation_split = 0.2)
            temp_weights = temp_classifier.get_weights()
            temp_classifier_runs_weights.append(deepcopy(temp_weights))
        level_2_classifiers_runs_weights.append(deepcopy(temp_classifier_runs_weights))

        del temp_X_train
        gc.collect()

    if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")
    
    level_2_num_labels = np.sum(category_lengths)
    level_2_category_indices = [0] * len(category_lengths)
    for i in range(1, len(level_2_category_indices)):
        level_2_category_indices[i] = level_2_category_indices[i - 1] + category_lengths[i - 1]

    level_2_labels = data.columns.values[level_1_num_labels:]
    level_2_other_labels_indices = [i for i, label in enumerate(level_2_labels) if '99' in label]

    print("Predicting")
    level_1_y_pred, level_2_y_pred, final_predictions = predict(base_level_1_classifier, level_1_classifier_runs_weights, level_2_classifiers, level_2_classifiers_runs_weights, level_2_num_labels, level_2_category_indices, \
                                                                level_2_other_labels_indices, X_test, setting)

    level_1_y_test, level_2_y_test, y_test = get_y_test(data, level_1_num_labels, level_2_other_labels_indices)

    print("Testing")
    save_results(level_1_y_pred, level_2_y_pred, final_predictions, level_1_y_test, level_2_y_test, y_test)
    print("Done")

def tune_model(X_train, y_train, tuner, epochs, setting):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)

    # Search hyperparameter space
    if not 'multi' in setting:
        tuner.search(X_train, y_train, epochs = epochs, validation_split = 0.2, callbacks=[stop_early])
    else:
        tuner.search([X_train, X_train, X_train], y_train, epochs = epochs, validation_split = 0.2, callbacks=[stop_early])

    # Fit model with optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    if not 'multi' in setting:
        history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.2, callbacks=[stop_early])
    else:
        history = model.fit([X_train, X_train, X_train], y_train, epochs = epochs, validation_split = 0.2, callbacks=[stop_early])

    # Find optimal epoch with lowest validation loss
    val_loss_per_epoch = history.history['val_loss']
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

    return model, best_epoch

class CNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes, setting):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.setting = setting

    def build(self, hp):
        hp_dropout = hp.Float('dropout', min_value = 0.1, max_value = 0.3, step = 0.1)
        hp_filter_sizes = hp.Int('filter_size', min_value = 100, max_value = 500, step = 100)
        if 'multi' in self.setting:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 4, step = 1)
        else:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 5, step = 1)

        inputs1 = Input(shape=self.input_shape)
        conv1 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel1_sizes, activation='relu')(inputs1)
        pool1 = GlobalMaxPool1D()(conv1)
        dropout1 = Dropout(hp_dropout)(pool1)
        flat1 = Flatten()(dropout1)

        # interpretation
        if not 'multi' in self.setting:
            outputs = Dense(self.num_classes, activation='softmax')(flat1)
            model = Model(inputs=[inputs1], outputs=outputs)
        else:
            hp_kernel2_sizes = hp.Int('kernel_size2', min_value = 5, max_value = 7, step = 1)
            hp_kernel3_sizes = hp.Int('kernel_size3', min_value = 7, max_value = 9, step = 1)

            inputs2 = Input(shape=self.input_shape)
            conv2 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel2_sizes, activation='relu')(inputs2)
            pool2 = GlobalMaxPool1D()(conv2)
            dropout2 = Dropout(hp_dropout)(pool2)
            flat2 = Flatten()(dropout2)

            inputs3 = Input(shape=self.input_shape)
            conv3 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel3_sizes, activation='relu')(inputs3)
            pool3 = GlobalMaxPool1D()(conv3)
            dropout3 = Dropout(hp_dropout)(pool3)
            flat3 = Flatten()(dropout3)

            merged = concatenate([flat1, flat2, flat3])
            outputs = Dense(self.num_classes, activation='softmax')(merged)
            model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
       
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        return model
   
    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Choice('batch_size', [32, 64, 128])
        return model.fit(*args, batch_size = batch_sizes, **kwargs)

class RNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes, setting):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        hp_lstm_units = hp.Int('lstm_units', min_value = 50, max_value = 300, step = 50)
        hp_dropout = hp.Float('dropout', min_value = 0.1, max_value = 0.3, step = 0.1)

        #Build model
        inputs = Input(shape=self.input_shape)
        lstm = LSTM(units=hp_lstm_units, activation='tanh')(inputs)
        dropout = Dropout(hp_dropout)(lstm)
        outputs = Dense(self.num_classes, activation='softmax')(dropout)
        model = Model(inputs=[inputs], outputs=outputs)

        #Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        return model
   
    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Choice('batch_size', [32, 64, 128])
        return model.fit(*args, batch_size = batch_sizes, **kwargs)

#----- Attention mechanisms -----#
class LabelwiseAttentionCNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes, setting):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.setting = setting

    def build(self, hp):

        hp_filter_sizes = hp.Int('filter_size', min_value = 100, max_value = 500, step = 100)
        if 'multi' in self.setting:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 4, step = 1)
        else:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 5, step = 1)

        # Build model
        inputs1 = Input(shape=self.input_shape)
        conv1 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel1_sizes, activation='relu')(inputs1)
        attention_layer1 = label_wise_attention(self.num_classes)(conv1)
       
        if self.setting == 'labelwise_attention':
            outputs = output_layer(self.num_classes)(attention_layer1)
            model = Model(inputs=[inputs1], outputs=outputs)
        elif self.setting == 'multi_labelwise_attention':
            hp_kernel2_sizes = hp.Int('kernel_size2', min_value = 5, max_value = 7, step = 1)
            hp_kernel3_sizes = hp.Int('kernel_size3', min_value = 7, max_value = 9, step = 1)

            inputs2 = Input(shape=self.input_shape)
            conv2 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel2_sizes, activation='relu')(inputs2)
            attention_layer2 = label_wise_attention(self.num_classes)(conv2)

            inputs3 = Input(shape=self.input_shape)
            conv3 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel3_sizes, activation='relu')(inputs3)
            attention_layer3 = label_wise_attention(self.num_classes)(conv3)
            attention_layers = concatenate([attention_layer1, attention_layer2, attention_layer3])

            outputs = output_layer(self.num_classes)(attention_layers)
            model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        #Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        return model
   
    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Choice('batch_size', [32, 64, 128])
        return model.fit(*args, batch_size = batch_sizes, **kwargs)

if __name__ == '__main__':
    main()