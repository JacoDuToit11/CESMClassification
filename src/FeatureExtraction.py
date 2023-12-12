#----- Obtains SciBERT word and document embeddings, performs fusion strategies -----#

# Imports
from category_descriptions import get_level_2_descriptions
from Data import get_dataset

import pandas as pd
import numpy as np
from npy_append_array import NpyAppendArray
import re
import time

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import os

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

from transformers import BertTokenizer, TFBertModel

from sklearn.model_selection import train_test_split

# Environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
pd.options.mode.chained_assignment = None
# tf.autograph.set_verbosity(0)
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Global variables
maxlen = 512
test_size = 0.15

# Driver for feature extraction and combination
def main():
    extract_bert_embeddings()
    bert_lda_words_combine()
    bert_lda_words_docs_combine()
    descriptions_bert_extraction()
    bert_lda_words_desc_combine()
    bert_lda_words_docs_desc_combine()

# Extracts the word embeddings and CLS token from last layour of BERT model
def extract_bert_embeddings():
    #Extract data
    print("---Feature Extraction Started---")
    tic = time.perf_counter()
    if extract_data:
        if os.path.exists("data/bertWordsTrain.npy"):
            os.remove("data/bertWordsTrain.npy")
            os.remove("data/CLStrain.npy")
            os.remove("data/bertWordsTest.npy")
            os.remove("data/CLStest.npy")

        #Obtain data
        X, y = get_dataset()

        #Split data into train/test sets
        X_train, X_test = train_test_split(X, test_size = test_size, random_state = 1000)

        bert_feature_extraction(X_train, 'train')
        bert_feature_extraction(X_test, 'test')

    toc = time.perf_counter()
    print(f'---Feature Extraction Completed in {toc - tic:0.4f} seconds')

def descriptions_bert_extraction():
    X = get_level_2_descriptions()
    if os.path.exists('data/bertWordsDescription.npy'):
            os.remove('data/bertWordsDescription.npy')
            os.remove('data/bertCLSDescription.npy')
    bert_feature_extraction(X, 'description')

def bert_feature_extraction(X, file_string):

    # SciBERT
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = TFBertModel.from_pretrained('allenai/scibert_scivocab_uncased', from_pt = True)

    #Obtain BERT output at batch size of 50
    index = 0
    inc = 50

    if file_string == 'train':
        file_name_words = 'data/bertWordsTrain.npy'
        file_name_CLS = 'data/CLStrain.npy'
    elif file_string == 'test':
        file_name_words = 'data/bertWordsTest.npy'
        file_name_CLS = 'data/CLStest.npy'
    elif file_string == 'description':
        file_name_words = 'data/bertWordsDescription.npy'
        file_name_CLS = 'data/CLSDescription.npy'
   
    CLS_file = NpyAppendArray(file_name_CLS)
    words_file = NpyAppendArray(file_name_words)
    while index + inc < len(X):
        temp = X[index:index + inc]
        inputs = tokenizer(list(temp), return_tensors="tf", padding=True, truncation=True, add_special_tokens=True, max_length=maxlen)
        outputs = model(inputs)

        #appending
        words_file.append(np.array(outputs.last_hidden_state))
        CLS_file.append(np.array(outputs.pooler_output))
        index += inc
        print("extracted ", index, " items", flush = True)

    if index != len(X):
        temp = X[index:len(X)]
        inputs = tokenizer(list(temp), return_tensors="tf", padding=True, truncation=True, add_special_tokens=True, max_length=maxlen)
        outputs = model(inputs)
        #appending
        words_file.append(np.array(outputs.last_hidden_state))
        CLS_file.append(np.array(outputs.pooler_output))
        print("extracted ", len(X), " items", flush = True)

def bert_lda_words_desc_combine():
    # Train data
    X_train = np.load("data/bertWordsDescription.npy", mmap_mode="r")
    ldaWords_train = np.load("data/ldaWordsDesc.npy", mmap_mode="r")
    if os.path.exists("data/bertAndLDAWordsDescription.npy"):
        os.remove("data/bertAndLDAWordsDescription.npy")
    train_file = NpyAppendArray("data/bertAndLDAWordsDescription.npy")
    for i in range(0, len(X_train)):
        temp_vec = np.zeros(shape=(512 + len(ldaWords_train[0]), 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_train[i, j]
        for j in range(512, 512 + len(ldaWords_train[0])):
            temp_vec[j] = ldaWords_train[i, j - 512]
        train_file.append(np.expand_dims(temp_vec, axis=0))

# Combines bert and lda word and document embeddings
def bert_lda_words_docs_desc_combine():
    # Train data
    X_train = np.load("data/bertWordsDescription.npy", mmap_mode="r")
    ldaWords_train = np.load("data/ldaWordsDesc.npy", mmap_mode="r")
    CLS_train = np.load("data/CLSDescription.npy", mmap_mode="r")
    lda_docs_train = np.load("data/ldaDocumentsDesc.npy", mmap_mode="r")

    if os.path.exists("data/allTogetherDescription.npy"):
        os.remove("data/allTogetherDescription.npy")
    train_file = NpyAppendArray("data/allTogetherDescription.npy")

    temp_vec_size = 512 + len(ldaWords_train[0]) + 2
    for i in range(0, len(X_train)):
        temp_vec = np.zeros(shape=(temp_vec_size, 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_train[i, j]
        for j in range(512, 512 + len(ldaWords_train[0])):
            temp_vec[j] = ldaWords_train[i, j - 512]
        temp_vec[temp_vec_size - 2] = CLS_train[i]
        temp_vec[temp_vec_size - 1] = lda_docs_train[i]

        train_file.append(np.expand_dims(temp_vec, axis=0))

# Combines bert and lda word embeddings
def bert_lda_words_combine():
    # Train data
    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    ldaWords_train = np.load("data/ldaWordsTrainBig.npy", mmap_mode="r")
    if os.path.exists("data/bertAndLDAWordsTrain.npy"):
        os.remove("data/bertAndLDAWordsTrain.npy")
    train_file = NpyAppendArray("data/bertAndLDAWordsTrain.npy")
    for i in range(0, len(X_train)):
        temp_vec = np.zeros(shape=(512 + len(ldaWords_train[0]), 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_train[i, j]
        for j in range(512, 512 + len(ldaWords_train[0])):
            temp_vec[j] = ldaWords_train[i, j - 512]
        train_file.append(np.expand_dims(temp_vec, axis=0))
   
    # Test data
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    ldaWords_test = np.load("data/ldaWordsTestBig.npy", mmap_mode="r")
    if os.path.exists("data/bertAndLDAWordsTest.npy"):
        os.remove("data/bertAndLDAWordsTest.npy")
    test_file = NpyAppendArray("data/bertAndLDAWordsTest.npy")
    for i in range(0, len(X_test)):
        temp_vec = np.zeros(shape=(512 + len(ldaWords_test[0]), 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_test[i, j]
        for j in range(512, 512 + len(ldaWords_test[0])):
            temp_vec[j] = ldaWords_test[i, j - 512]
        test_file.append(np.expand_dims(temp_vec, axis=0))

# Combines bert and lda word and document embeddings
def bert_lda_words_docs_combine():
    # Train data
    X_train = np.load("data/bertWordsTrain.npy", mmap_mode="r")
    ldaWords_train = np.load("data/ldaWordsTrainBig.npy", mmap_mode="r")
    CLS_train = np.load("data/CLStrain.npy", mmap_mode="r")
    lda_docs_train = np.load("data/ldaDocumentsTrainBig.npy", mmap_mode="r")

    if os.path.exists("data/allTogetherTrain.npy"):
        os.remove("data/allTogetherTrain.npy")
    train_file = NpyAppendArray("data/allTogetherTrain.npy")

    temp_vec_size = 512 + len(ldaWords_train[0]) + 2
    for i in range(0, len(X_train)):
        temp_vec = np.zeros(shape=(temp_vec_size, 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_train[i, j]
        for j in range(512, 512 + len(ldaWords_train[0])):
            temp_vec[j] = ldaWords_train[i, j - 512]
        temp_vec[temp_vec_size - 2] = CLS_train[i]
        temp_vec[temp_vec_size - 1] = lda_docs_train[i]

        train_file.append(np.expand_dims(temp_vec, axis=0))
   
    # Test data
    X_test = np.load("data/bertWordsTest.npy", mmap_mode="r")
    ldaWords_test = np.load("data/ldaWordsTestBig.npy", mmap_mode="r")
    CLS_test = np.load("data/CLStest.npy", mmap_mode="r")
    lda_docs_test = np.load("data/ldaDocumentsTestBig.npy", mmap_mode="r")

    if os.path.exists("data/allTogetherTest.npy"):
        os.remove("data/allTogetherTest.npy")
    test_file = NpyAppendArray("data/allTogetherTest.npy")
    for i in range(0, len(X_test)):
        temp_vec = np.zeros(shape=(temp_vec_size, 768), dtype="float32")
        for j in range(0, 512):
            temp_vec[j] = X_test[i, j]
        for j in range(512, 512 + len(ldaWords_test[0])):
            temp_vec[j] = ldaWords_test[i, j - 512]
        temp_vec[temp_vec_size - 2] = CLS_test[i]
        temp_vec[temp_vec_size - 1] = lda_docs_test[i]
        test_file.append(np.expand_dims(temp_vec, axis=0))

if __name__ == '__main__':
    main()