#----- Obtains LDA word and document embeddings -----#

import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import re

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import ldamodel

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split

from category_descriptions import get_level_2_descriptions

mallet_path = 'mallet-2.0.8/bin/mallet'

lda_iter = 1000
num_tops = 50
dim_size = 768
num_words = 512

test_size = 0.15

def main():
    extractLDA()

def extractLDA():

    if os.path.exists("data/ldaDocumentsTrainBig.npy"):
        os.remove("data/ldaDocumentsTrainBig.npy")
        os.remove("data/ldaDocumentsTestBig.npy")
        os.remove("data/ldaWordsTrainBig.npy")
        os.remove("data/ldaWordsTestBig.npy")
        os.remove("data/ldaDocumentsDesc.npy")
        os.remove("data/ldaWordsDesc.npy")

    X, num_desc = get_dataset()

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    data = list(sent_to_words(X))

    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus
    texts = data

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_tops, id2word=id2word, iterations=lda_iter)
    
    # Obtain document topic distributions
    Z = np.zeros(shape=(len(texts), dim_size), dtype="float32")
    j = 0
    for doc_topics in lda_mallet[corpus]:
        topic_vec = np.append([doc_topics[i][1] for i in range(num_tops)], np.zeros(shape = (dim_size - num_tops)))
        Z[j] = topic_vec
        j += 1
    
    # Obtain word topic distributions
    word_freq_matrix = lda_mallet.load_word_topics()
    totals = np.sum(word_freq_matrix, axis = 0)
    word_matrix = np.zeros(shape = (len(word_freq_matrix), len(word_freq_matrix[0])))

    for i in range(len(word_freq_matrix)):
        for j in range(len(word_freq_matrix[0])):
            if (totals[j] != 0):
                word_matrix[i][j] = word_freq_matrix[i][j] / totals[j]

    W = np.zeros(shape=(len(texts), num_words, dim_size), dtype="float32")
    for i in range(0, len(texts)):
        for j in range(0, min(len(texts[i]), num_words)):
            word_index = id2word.doc2idx([texts[i][j]])[0]
            W[i, j] = np.append(word_matrix[:, word_index], np.zeros(shape = (dim_size - num_tops)))
    
    # Save data to files
    desc_Z = Z[len(Z)-num_desc:]
    np.save("data/ldaDocumentsDesc.npy", desc_Z)
    Z = Z[0:len(Z)-num_desc]
    ldaDocs_train, ldaDocs_test= train_test_split(Z, test_size = test_size, random_state = 1000)
    np.save("data/ldaDocumentsTrainBig.npy", ldaDocs_train)
    np.save("data/ldaDocumentsTestBig.npy", ldaDocs_test)

    desc_W = W[len(W)-num_desc:]
    np.save("data/ldaWordsDesc.npy", desc_W)
    W = W[0:len(W)-num_desc]
    ldaWords_train, ldaWords_test= train_test_split(W, test_size = test_size, random_state = 1000)
    np.save("data/ldaWordsTrainBig.npy", ldaWords_train)
    np.save("data/ldaWordsTestBig.npy", ldaWords_test)
    
def get_dataset():
    data = pd.read_csv('data/data.csv')

    # Preprocessing
    def lowercase(sentence):
        sentence = sentence.lower()
        return sentence

    def decontract(sentence):
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence
    
    def removePunctuation(sentence): 
        sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        sentence = re.sub(r'[.|,|;|)|(|\|/]',r' ',sentence)
        sentence = sentence.strip()
        sentence = sentence.replace("\n"," ")
        return sentence
    
    def removeStopWords(sentence):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        return pattern.sub('', sentence)
    
    def stemming(sentence):
        stemmer = SnowballStemmer("english")
        stemmedSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemmedSentence += stem
            stemmedSentence += " "
        stemmedSentence = stemmedSentence.strip()
        return stemmedSentence

    # Appending textual descriptions
    content = data['content'].values
    level_2_desc = get_level_2_descriptions()
    num_desc = len(level_2_desc)
    content = np.append(content, level_2_desc)

    total_words = 0
    for i in range(0, len(content)):
        content[i] = lowercase(content[i])
        content[i] = decontract(content[i])
        content[i] = removePunctuation(content[i])
        content[i] = removeStopWords(content[i])
        content[i] = stemming(content[i])
        total_words += len(content[i].split(' '))
        print(len(content[i].split(' ')))
    
    return content, num_desc

if __name__ == '__main__':
    main()