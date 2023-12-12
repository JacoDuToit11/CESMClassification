#----- Embedding attention mechanism -----#

# Imports
from category_descriptions import get_level_2_descriptions
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

# Environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
pd.options.mode.chained_assignment = None
# tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Global variables
maxlen = 512
test_size = 0.15
level_2_thresholds = [89, 200, 170, 110, 160, 95, 145, 130, 73, 70, 130]

num_runs = 10

# Driver for the different model types
def main():

    global configs, file_name

    setting = "s1_embedding_attention"
    file_name = "Results/S1/" + setting + "_results.txt"
    configs = bert_words_embedding_attention_tuning(setting)
    combined_level_1_y_pred = []
    combined_level_2_y_pred = []
    combined_final_predictions = []
    for i in range(num_runs):
        level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = bert_words_cnn_model(setting)
        combined_level_1_y_pred.append(level_1_y_pred)
        combined_level_2_y_pred.append(level_2_y_pred)
        combined_final_predictions.append(final_predictions)
    save_results(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test)

    setting = "s2_embedding_attention"
    file_name = "Results/S2/" + setting + "_results.txt"
    configs = bert_lda_embedding_attention_tuning(setting)
    combined_level_1_y_pred = []
    combined_level_2_y_pred = []
    combined_final_predictions = []
    for i in range(num_runs):
        level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = bert_lda_words_cnn_model(setting)
        combined_level_1_y_pred.append(level_1_y_pred)
        combined_level_2_y_pred.append(level_2_y_pred)
        combined_final_predictions.append(final_predictions)
    save_results(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test)

    setting = "s3_embedding_attention"
    file_name = "Results/S3/" + setting + "_results.txt"
    configs = bert_lda_words_docs_embedding_attention_tuning(setting)
    combined_level_1_y_pred = []
    combined_level_2_y_pred = []
    combined_final_predictions = []
    for i in range(num_runs):
        level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = bert_lda_words_docs_cnn_model(setting)
        combined_level_1_y_pred.append(level_1_y_pred)
        combined_level_2_y_pred.append(level_2_y_pred)
        combined_final_predictions.append(final_predictions)
    save_results(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test)

# Classifies papers by using BERT word embeddings as input to CNN
def bert_words_embedding_attention_tuning(setting):

    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    configs = embedding_attention_tuning(X_train, X_test, setting)
    return configs

def bert_lda_embedding_attention_tuning(setting):

    X_train = np.load("data/bertAndLDAWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertAndLDAWordsTest.npy", mmap_mode="r")
    configs = embedding_attention_tuning(X_train, X_test, setting)
    return configs

# Classifies papers by using BERT and LDA word embeddings and document vectors as input to CNN
def bert_lda_words_docs_embedding_attention_tuning(setting):

    X_train = np.load("data/allTogetherTrain.npy", mmap_mode="r")
    X_test = np.load("data/allTogetherTest.npy", mmap_mode="r")
    configs = embedding_attention_tuning(X_train, X_test, setting)
    return configs

# Classifies papers by using BERT word embeddings as input to CNN
def bert_words_cnn_model(setting):

    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = classify(X_train, X_test, setting)
    return level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions

def bert_lda_words_cnn_model(setting):

    X_train = np.load("data/bertAndLDAWordsTrain.npy", mmap_mode="r")
    X_test = np.load("data/bertAndLDAWordsTest.npy", mmap_mode="r")
    level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = classify(X_train, X_test, setting)
    return level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions

# Classifies papers by using BERT and LDA word embeddings and document vectors as input to CNN
def bert_lda_words_docs_cnn_model(setting):

    X_train = np.load("data/allTogetherTrain.npy", mmap_mode="r")
    X_test = np.load("data/allTogetherTest.npy", mmap_mode="r")
    level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions = classify(X_train, X_test, setting)
    return level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions

def embedding_attention_tuning(X_train, X_test, setting):
    placeholder, data = get_dataset()

    if "s1" in setting:
        query_matrix = np.load("data/bertWordsDescription.npy", mmap_mode="r")
    elif "s2" in setting:
        query_matrix = np.load("data/bertAndLDAWordsDescription.npy", mmap_mode="r")
    elif "s3" in setting:
        query_matrix = np.load("data/allTogetherDescription.npy", mmap_mode="r")

    level_1_hyper_model = CNNHyperModel
    level_2_hyper_model = LabelEmbeddingAttentionCNNHyperModel
    new_query_matrix = []
    new_query_matrix.append(query_matrix[0:2])
    new_query_matrix.append(query_matrix[2:4])
    new_query_matrix.append(query_matrix[4:6])
    new_query_matrix.append(query_matrix[6:9])
    new_query_matrix.append(query_matrix[9:11])
    new_query_matrix.append(query_matrix[11:13])
    new_query_matrix.append(query_matrix[13:16])
    new_query_matrix.append(query_matrix[16:20])
    new_query_matrix.append(query_matrix[20:22])
    new_query_matrix.append(query_matrix[22:24])
    new_query_matrix.append(query_matrix[24:28])

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

    max_epochs = 10
    factor = 3
    epochs = 10

    hyperparameter_configs = []
    print("ROOT CLASSIFIER")
    tuner = kt.Hyperband(level_1_hyper_model(input_shape, num_classes, setting), 
                        objective = 'val_loss', 
                        max_epochs = max_epochs, 
                        factor = factor, 
                        project_name = 'keras_tuner_file',
                        overwrite = True)
    
    tune_model(level_1_X, level_1_y, tuner, epochs, setting, hyperparameter_configs)

    # Build classifiers using data from subcategories in level 2
    level_2_start_labels = ['101', '401', '702', '806', '907', '1101', '1302', '1404', '1702', '1808', '2003']
    level_2_end_labels = ['199', '499', '799', '899', '999', '1199', '1399', '1499', '1799', '1899', '2099']
    num_level_2_classifiers = len(level_2_start_labels)

    max_epochs = 30
    factor = 3
    epochs = 30

    category_lengths = []
    for i in range(num_level_2_classifiers):
        print("CLASSIFIER ", i)
        level_2_data = data.loc[:, level_2_start_labels[i]:level_2_end_labels[i]]
        level_2_data = level_2_data.iloc[:, 0:-1]
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
        temp_num_classes = temp_y_train.shape[1]
        tuner = kt.Hyperband(level_2_hyper_model(input_shape, temp_num_classes, new_query_matrix[i], setting), 
                            objective = 'val_loss', 
                            max_epochs = max_epochs, 
                            factor = factor, 
                            project_name = 'keras_tuner_file',
                            overwrite = True)

        tune_model(temp_X_train, temp_y_train, tuner, epochs, setting, hyperparameter_configs)
        del temp_X_train
        gc.collect()

    print('Level 2 configs: ')
    print(hyperparameter_configs)
    return hyperparameter_configs

def tune_model(X_train, y_train, tuner, epochs, setting, hyperparameter_configs):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3)

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
    temp_config = [best_hps.get('filter_size'), best_hps.get('kernel_size1'), best_hps.get('batch_size'), best_epoch]
    hyperparameter_configs.append(temp_config)
    print('Filter size: ', temp_config[0])
    print('Kernel size: ', temp_config[1])
    print('Batch size: ', temp_config[2])
    print('Best epoch: ', best_epoch)
    model = tuner.hypermodel.build(best_hps)

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
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 8, step = 2)

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

class LabelEmbeddingAttentionCNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes, query_matrix, setting):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.setting = setting
        self.query_matrix = query_matrix

    def build(self, hp):
        hp_filter_sizes = hp.Int('filter_size', min_value = 100, max_value = 500, step = 100)
        if not 'multi' in self.setting:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 6, step = 2)
        else:
            hp_kernel1_sizes = hp.Int('kernel_size1', min_value = 2, max_value = 3, step = 1)

        # Build model
        inputs1 = Input(shape=self.input_shape)
        conv1 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel1_sizes, activation='relu')(inputs1)
        attention_layer1 = general_attention()([conv1, self.query_matrix])

        if not 'multi' in self.setting:
            outputs = output_layer(self.num_classes)(attention_layer1)
            model = Model(inputs=[inputs1], outputs=outputs)

        else:
            hp_kernel2_sizes = hp.Int('kernel_size2', min_value = 4, max_value = 5, step = 1)
            hp_kernel3_sizes = hp.Int('kernel_size3', min_value = 6, max_value = 8, step = 2)

            inputs2 = Input(shape=self.input_shape)
            conv2 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel2_sizes, activation='relu')(inputs2)
            attention_layer2 = general_attention()([conv2, self.query_matrix])

            inputs3 = Input(shape=self.input_shape)
            conv3 = Conv1D(filters=hp_filter_sizes, kernel_size = hp_kernel3_sizes, activation='relu')(inputs3)
            attention_layer3 = general_attention()([conv3, self.query_matrix])

            attention_layers = concatenate([attention_layer1, attention_layer2, attention_layer3])
            outputs = output_layer(self.num_classes)(attention_layers)
            model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        #Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        return model
   
    def fit(self, hp, model, *args, **kwargs):
        batch_sizes = hp.Choice('batch_size', [32, 64, 128])
        return model.fit(*args, batch_size = batch_sizes, **kwargs)

def classify(X_train, X_test, setting):
    placeholder, data = get_dataset()

    root_classifier = cnn_model
    level_2_classifier = label_embedding_attention_cnn_model
    if "s1" in setting:
        query_matrix = np.load("data/bertWordsDescription.npy", mmap_mode="r")
    elif "s2" in setting:
        query_matrix = np.load("data/bertAndLDAWordsDescription.npy", mmap_mode="r")
    elif "s3" in setting:
        query_matrix = np.load("data/allTogetherDescription.npy", mmap_mode="r")
    new_query_matrix = []
    new_query_matrix.append(query_matrix[0:2])
    new_query_matrix.append(query_matrix[2:4])
    new_query_matrix.append(query_matrix[4:6])
    new_query_matrix.append(query_matrix[6:9])
    new_query_matrix.append(query_matrix[9:11])
    new_query_matrix.append(query_matrix[11:13])
    new_query_matrix.append(query_matrix[13:16])
    new_query_matrix.append(query_matrix[16:20])
    new_query_matrix.append(query_matrix[20:22])
    new_query_matrix.append(query_matrix[22:24])
    new_query_matrix.append(query_matrix[24:28])
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
    level_1_classifier = root_classifier(level_1_X, level_1_y, setting, configs[0])
    del level_1_X
    gc.collect()
    if os.path.exists("data/temp_file.npy"):
            os.remove("data/temp_file.npy")

    # Build classifiers using data from subcategories in level 2
    level_2_start_labels = ['101', '401', '702', '806', '907', '1101', '1302', '1404', '1702', '1808', '2003']
    level_2_end_labels = ['199', '499', '799', '899', '999', '1199', '1399', '1499', '1799', '1899', '2099']

    category_lengths = []
    level_2_classifiers = []
    for i in range(len(level_2_start_labels)):
        level_2_data = data.loc[:, level_2_start_labels[i]:level_2_end_labels[i]]
        level_2_data = level_2_data.iloc[:, 0:-1]

        level_2_labels = level_2_data.to_numpy()
        category_lengths.append(level_2_labels.shape[1])
        y_train, y_test = train_test_split(level_2_labels, test_size = test_size, random_state = 1000)
        temp_X_train, temp_y_train = get_data_for_node(X_train, y_train, level_2_thresholds[i])
        level_2_classifiers.append(level_2_classifier(temp_X_train, temp_y_train, new_query_matrix[i], setting, (i+1)))

    for i in range(len(category_lengths)):
        category_lengths[i] += 1

    level_2_num_labels = sum(category_lengths)

    level_2_num_labels = np.sum(category_lengths)
    level_2_category_indices = [0] * len(category_lengths)
    for i in range(1, len(level_2_category_indices)):
        level_2_category_indices[i] = level_2_category_indices[i - 1] + category_lengths[i - 1]

    level_2_labels = data.columns.values[level_1_num_labels:]
    level_2_other_labels_indices = [i for i, label in enumerate(level_2_labels) if '99' in label]

    level_1_y_pred, level_2_y_pred, final_predictions = normal_predict(level_1_classifier, level_2_classifiers,  level_2_num_labels, level_2_category_indices, \
                                                                level_2_other_labels_indices, X_test, setting)

    level_1_y_test, level_2_y_test, y_test = get_y_test(data, level_1_num_labels, level_2_other_labels_indices)

    return level_1_y_test, level_1_y_pred, level_2_y_test, level_2_y_pred, y_test, final_predictions

# CNN model used to classify research outputs
def cnn_model(X_train, y_train, setting, config):
    shape = X_train.shape[1:]

    filter_size = config[0]
    kernel_size_1 = config[1]
    batch_size = config[2]
    epochs = config[3]

    #Build model
    inputs1 = Input(shape=shape)
    conv1 = Conv1D(filters=filter_size, kernel_size = kernel_size_1, activation='relu')(inputs1)
    pool1 = GlobalMaxPool1D()(conv1)
    dropout1 = Dropout(0.1)(pool1)
    flat1 = Flatten()(dropout1)
   
    if not 'multi' in setting:
        outputs = Dense(y_train.shape[1], activation='softmax')(flat1)
        model = Model(inputs=[inputs1], outputs=outputs)
    else:
        kernel_size_2 = 4
        kernel_size_3 = 6

        inputs2 = Input(shape=shape)
        conv2 = Conv1D(filters=filter_size, kernel_size = kernel_size_2, activation='relu')(inputs2)
        pool2 = GlobalMaxPool1D()(conv2)
        flat2 = Flatten()(pool2)

        inputs3 = Input(shape=shape)
        conv3 = Conv1D(filters=filter_size, kernel_size = kernel_size_3, activation='relu')(inputs3)
        pool3 = GlobalMaxPool1D()(conv3)
        flat3 = Flatten()(pool3)

        merged = concatenate([flat1, flat2, flat3])

        outputs = Dense(y_train.shape[1], activation='softmax')(merged)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    #Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    print(model.summary())

    #Train model with data
    if not 'multi' in setting:
        model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2)
    else:
        model.fit([X_train, X_train, X_train], y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2)
    return model

def label_embedding_attention_cnn_model(X_train, y_train, query_matrix, setting, classifier_index):
    shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    filter_size = configs[classifier_index][0]
    kernel_size_1 = configs[classifier_index][1]
    batch_size = configs[classifier_index][2]
    epochs = configs[classifier_index][3]

    # Build model
    inputs1 = Input(shape=shape)
    conv1 = Conv1D(filters=filter_size, kernel_size = kernel_size_1, activation='relu')(inputs1)
    attention_layer1 = general_attention()([conv1, query_matrix])

    if not 'multi' in setting:
        outputs = output_layer(num_classes)(attention_layer1)
        model = Model(inputs=[inputs1], outputs=outputs)
    else:
        kernel_size_2 = 4
        kernel_size_3 = 6

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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    #Train model with data
    if not 'multi' in setting:
        model.fit([X_train], y_train, epochs = epochs, batch_size = batch_size)
    else:
        model.fit([X_train, X_train, X_train], y_train, epochs = epochs, batch_size = batch_size)
    return model

if __name__ == '__main__':
    main()