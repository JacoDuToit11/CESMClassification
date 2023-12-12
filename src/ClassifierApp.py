#----- Classifier application -----#

import os

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import re

import tensorflow as tf
import transformers
from transformers import BertTokenizer, AutoTokenizer, TFBertModel, pipeline
from transformers import TFBertModel, BertConfig, BertTokenizerFast
from transformers import AlbertTokenizer, TFAlbertModel, AlbertConfig
transformers.logging.set_verbosity_error()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tika import parser
import fitz

max_words = 100000
maxlen = 512

def main():
    GUI()

def commandLineOutput():
    #file_name = "testTheses/testPaperEngineering2.txt"
    file_name = "testTheses/testPaperTheology.txt"
    #file_name = "testTheses/testPaperEngineering.txt"
    #file_name = "testTheses/testPaperSocialSciences.txt"
    #file_name = "testTheses/testPaperEducation.txt"
    #file_name = "testTheses/testPaperLinguistics.txt"
    #file_name = "testTheses/testPaperAgriculture.txt"

    if file_name.endswith(".pdf"):
        with fitz.open(file_name) as doc:
            test_paper = ""
            for page in doc:
                test_paper += page.get_text()
    else:
        with open(file_name) as f:
            test_paper = f.readlines()

    model = tf.keras.models.load_model('saved_models/bert_words_model')
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    bert_model = TFAlbertModel.from_pretrained("albert-base-v2")
    inputs = tokenizer(test_paper, return_tensors="tf", padding="max_length", truncation=True, add_special_tokens=True, max_length=maxlen)
    outputs = bert_model(inputs)
    test_paper = np.array(outputs.last_hidden_state)
    y_pred = model.predict(test_paper)

    print("-------------------RESULTS-------------------")
    print("Model output: ", y_pred)
    y_pred = np.array(list(y_pred.round()))   
    print()
    print("Label result: ", y_pred)

    with open ('labels.txt', 'rb') as fp:
        labels = pickle.load(fp)

    results = ''
    for i in range(0, len(y_pred)):
        print()
        print("This thesis belongs to these CESM categories:")
        for j in range(0, len(labels)):
            if(y_pred[i][j] == 1):
                print(labels[j])
                results += labels[j] + '\n'

def getResults(file_name):
    if file_name.endswith(".pdf"):
        with fitz.open(file_name) as doc:
            test_paper = ""
            for page in doc:
                test_paper += page.get_text()
    else:
        with open(file_name) as f:
            test_paper = f.readlines()


    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    bert_model = TFAlbertModel.from_pretrained("albert-base-v2")

    inputs = tokenizer(test_paper, return_tensors="tf", padding="max_length", truncation=True, add_special_tokens=True, max_length=maxlen)

    outputs = bert_model(inputs)

    test_paper = np.array(outputs.last_hidden_state)

    root_classifier = tf.keras.models.load_model('saved_models/root_model')
    level_1_y_pred = root_classifier.predict(test_paper)
    level_2_y_pred = []
    threshold = 0.8

    level_1_classifiers = []
    for i in range(1, 12):
        level_1_classifiers.append(tf.keras.models.load_model('saved_models/level_1_model_' + str(i)))

    final_predictions = []
    # Get the predicted level 1 output
    level_1_max_index = np.where(level_1_y_pred[0] == np.amax(level_1_y_pred[0]))[0][0]
    level_1_y_pred[0] = [0] * len(level_1_y_pred[0])
    level_1_y_pred[0][level_1_max_index] = 1
    level_1_classifier = level_1_classifiers[level_1_max_index]
    level_2_y_pred.append(np.squeeze(level_1_classifier.predict(test_paper)))

    category_lengths = [2, 2, 2, 3, 3, 2, 3, 4, 3, 2, 4]
    level_2_num_labels = np.sum(category_lengths)
    level_2_category_indices = [0] * len(category_lengths)
    for i in range(1, len(level_2_category_indices)):
        level_2_category_indices[i] = level_2_category_indices[i - 1] + category_lengths[i - 1]

    # Get the predicted level 2 output by only looking at subcategory of level 1 output
    level_2_max_index = np.where(level_2_y_pred[0] == np.amax(level_2_y_pred[0]))[0][0]
    level_2_confidence = level_2_y_pred[0][level_2_max_index]
    level_2_y_pred[0] = [0] * level_2_num_labels
    if (level_2_confidence >= threshold):
        level_2_y_pred[0][level_2_category_indices[level_1_max_index] + level_2_max_index] = 1
    final_predictions.append([*level_1_y_pred[0], *level_2_y_pred[0]])

    final_predictions = np.array(final_predictions)

    print("Model output: ", final_predictions)
    final_predictions = list(final_predictions)   

    # level_1_labels = ['Agriculture, Agricultural Operations and Related Sciences', 'Business, Economics and Management Studies', 'Education', 'Engineering', 'Health Professions and Related Clinical Sciences', 
    # 'Languages, Linguistics and Literature', 'Life Sciences', 'Physical Sciences', 'Philosophy, Religion and Theology', 'Psychology', 'Social Sciences']

    # level_2_labels = ['Agricultural Business and Management', 'Plant Sciences', 'Business Administration, Management and Operations', 'Economics', 'Curriculum and Instruction', 
    # 'Educational Management and Leadership', 'Chemical Engineering', 'Electrical, Electronics and Communications Engineering', 'Mechanical and Mechatronic Engineering',
    # 'Medical Clinical Sciences', 'Nursing', 'Public Health', 'Linguistic, Comparative and Related Language Studies and Practices', 'English Language and Literature', 'Biochemistry, Biophysics and Molecular Biochemistry',
    # 'Botany/Plant Biology', 'Zoology/Animal Biology', 'Chemistry', 'Geography and Cartography', 'Geology and Earth Sciences/Geosciences', 'Physics', 'Philosophy',
    # 'Religion', 'Theology', 'Educational Psychology', 'Industrial and Organisational Psychology', 'History', 'Political Science and Government', 'Sociology', 'Social Work']

    all_labels = ['Agriculture, Agricultural Operations and Related Sciences', 'Business, Economics and Management Studies', 'Education', 'Engineering', 'Health Professions and Related Clinical Sciences', 
    'Languages, Linguistics and Literature', 'Life Sciences', 'Physical Sciences', 'Philosophy, Religion and Theology', 'Psychology', 'Social Sciences', 'Agricultural Business and Management', 'Plant Sciences', 
    'Business Administration, Management and Operations', 'Economics', 'Curriculum and Instruction', 
    'Educational Management and Leadership', 'Chemical Engineering', 'Electrical, Electronics and Communications Engineering', 'Mechanical and Mechatronic Engineering',
    'Medical Clinical Sciences', 'Nursing', 'Public Health', 'Linguistic, Comparative and Related Language Studies and Practices', 'English Language and Literature', 'Biochemistry, Biophysics and Molecular Biochemistry',
    'Botany/Plant Biology', 'Zoology/Animal Biology', 'Chemistry', 'Geography and Cartography', 'Geology and Earth Sciences/Geosciences', 'Physics', 'Philosophy',
    'Religion', 'Theology', 'Educational Psychology', 'Industrial and Organisational Psychology', 'History', 'Political Science and Government', 'Sociology', 'Social Work']

    results = ''
    for i in range(0, len(final_predictions)):
        for j in range(0, len(all_labels)):
            if(final_predictions[i][j] == 1):
                results += all_labels[j] + '\n'
    print()
    print("This thesis belongs to these CESM categories:")
    print(results)
    return results

if __name__ == '__main__':
    main()