#----- Utility functions -----#

#----- Imports -----#
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

# Prediction related functions
def predict(base_level_1_classifier, level_1_classifier_runs_weights, level_2_classifiers, level_2_classifiers_runs_weights, level_2_num_labels, level_2_category_indices, \
                                                                level_2_other_labels_indices, X_test, setting):

    combined_level_1_y_pred = []
    combined_level_2_y_pred = []
    combined_final_predictions = []
    for run in range(len(level_1_classifier_runs_weights)):
        level_1_weights = level_1_classifier_runs_weights[run]
        level_1_classifier = clone_model(base_level_1_classifier)
        level_1_classifier.compile(optimizer='adam', loss='categorical_crossentropy')
        level_1_classifier.set_weights(level_1_weights)
        # Predict level 1   
        if not 'multi' in setting:
            level_1_y_pred = level_1_classifier.predict([X_test])
        else:
            level_1_y_pred = level_1_classifier.predict([X_test, X_test, X_test])

        level_2_confidence_threshold = 0.8
        final_predictions = []
        level_2_y_pred = []

        # Load level 2 classifiers
        level_2_classifiers_weights = [classifier_weights[run] for classifier_weights in level_2_classifiers_runs_weights]

        for i in range(len(level_2_category_indices)):
            level_2_weights = level_2_classifiers_weights[i]
            level_2_classifiers[i].set_weights(level_2_weights)

        print("Predicting level 2")
        for i in range(0,  X_test.shape[0]):
            print("Predicting test ", i)
            # Get the predicted level 1 output
            level_1_max_index = np.where(level_1_y_pred[i] == np.amax(level_1_y_pred[i]))[0][0]
            level_1_y_pred[i] = [0] * len(level_1_y_pred[i])
            level_1_y_pred[i][level_1_max_index] = 1

            level_2_classifier = level_2_classifiers[level_1_max_index]

            if setting in ['single', 'labelwise_attention', 'embedding_attention', 'rnn']:
                temp_level_2_y_pred = np.squeeze(level_2_classifier.predict([X_test[i:i+1]]))
            elif setting in ['multi', 'multi_labelwise_attention', 'multi_embedding_attention']:
                temp_level_2_y_pred = np.squeeze(level_2_classifier.predict([X_test[i:i+1], X_test[i:i+1], X_test[i:i+1]]))

            # Get the predicted level 2 output by only looking at subcategory of level 1 output
            level_2_max_index = np.where(temp_level_2_y_pred == np.amax(temp_level_2_y_pred))[0][0]
            level_2_confidence = temp_level_2_y_pred[level_2_max_index]
            level_2_y_pred.append([0] * level_2_num_labels)

            if (level_2_confidence >= level_2_confidence_threshold):
                level_2_y_pred[i][level_2_category_indices[level_1_max_index] + level_2_max_index] = 1

            for j in level_2_other_labels_indices:
                level_2_y_pred[i][j] = 0
            final_predictions.append([*level_1_y_pred[i], *level_2_y_pred[i]])
        
        level_1_y_pred = np.array(level_1_y_pred)
        level_2_y_pred = np.array(level_2_y_pred)
        final_predictions = np.array(final_predictions)

        combined_level_1_y_pred.append(deepcopy(level_1_y_pred))
        combined_level_2_y_pred.append(deepcopy(level_2_y_pred))
        combined_final_predictions.append(deepcopy(final_predictions))

    return combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions

def normal_predict(level_1_classifier, level_2_classifiers,  level_2_num_labels, level_2_category_indices, \
                                                                level_2_other_labels_indices, X_test, setting):

    # Predict level 1   
    if not 'multi' in setting:
        level_1_y_pred = level_1_classifier.predict([X_test])
    else:
        level_1_y_pred = level_1_classifier.predict([X_test, X_test, X_test])

    level_2_confidence_threshold = 0.8
    final_predictions = []
    level_2_y_pred = []
    print("Predicting level 2")
    for i in range(X_test.shape[0]):
        print("Predicting test ", i)
        # Get the predicted level 1 output
        level_1_max_index = np.where(level_1_y_pred[i] == np.amax(level_1_y_pred[i]))[0][0]
        level_1_y_pred[i] = [0] * len(level_1_y_pred[i])
        level_1_y_pred[i][level_1_max_index] = 1

        level_2_classifier = level_2_classifiers[level_1_max_index]

        if setting in ['single', 'labelwise_attention', 'embedding_attention', 'rnn']:
            temp_level_2_y_pred = np.squeeze(level_2_classifier.predict([X_test[i:i+1]]))
        elif setting in ['multi', 'multi_labelwise_attention', 'multi_embedding_attention']:
            temp_level_2_y_pred = np.squeeze(level_2_classifier.predict([X_test[i:i+1], X_test[i:i+1], X_test[i:i+1]]))

        # Get the predicted level 2 output by only looking at subcategory of level 1 output
        level_2_max_index = np.where(temp_level_2_y_pred == np.amax(temp_level_2_y_pred))[0][0]
        level_2_confidence = temp_level_2_y_pred[level_2_max_index]
        level_2_y_pred.append([0] * level_2_num_labels)

        if (level_2_confidence >= level_2_confidence_threshold):
            level_2_y_pred[i][level_2_category_indices[level_1_max_index] + level_2_max_index] = 1

        for j in level_2_other_labels_indices:
            level_2_y_pred[i][j] = 0
        final_predictions.append([*level_1_y_pred[i], *level_2_y_pred[i]])
    
    level_1_y_pred = np.array(level_1_y_pred)
    level_2_y_pred = np.array(level_2_y_pred)
    final_predictions = np.array(final_predictions)

    return level_1_y_pred, level_2_y_pred, final_predictions

def get_y_test(data, level_1_num_labels, level_2_other_labels_indices):
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
    level_2_y_test = np.array(level_2_y_test)

    return level_1_y_test, level_2_y_test, y_test

#----- Data related functions -----#
# Obtains the training data required for a classifier by only adding entries up to a certain threshold to ensure a balanced dataset
def get_data_for_node(X_train, y_train, threshold):
    X_new = []
    y_new = []
    counts = [0] * y_train.shape[1]
    for i in range(X_train.shape[0]):
        for j in range(y_train.shape[1]):
            if counts[j] < threshold:
                # Only add first 'threshold' entries of each category
                if y_train[i][j] == 1:
                    counts[j] += 1
                    X_new.append(X_train[i])
                    y_new.append(y_train[i])
    return np.array(X_new), np.array(y_new)

# Tokenizes corpus of document by assinging an integer to each word. A document is thus represented by a sequence of integers.
def basicTokenize(X):
    #Tokenize data
    tokenizer = Tokenizer(num_words = 5000, lower=True)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    # X = pad_sequences(sequences, maxlen=maxlen, padding ="post", truncating="post")
    X = pad_sequences(sequences, maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1
    return X, vocab_size