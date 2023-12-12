#----- Evaluation functions -----#

# Imports
import numpy as np
import os

def save_results(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test, file_name): 
    if os.path.exists(file_name):
            os.remove(file_name)
    f = open(file_name, "w")

    print("MAIN RESULTS:", file = f)
    evaluate_model_combined(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test, f)
    print(file = f)
    
    for i in range(len(combined_final_predictions)):
        print('RUN NUMBER ', i + 1, file = f)
        level_1_y_pred = combined_level_1_y_pred[i]
        level_2_y_pred = combined_level_2_y_pred[i]
        final_predictions = combined_final_predictions[i]

        print("Level 1 RESULTS:", file = f)
        evaluate_model(level_1_y_pred, level_1_y_test, f, 1)
        print(file = f)
        print("Level 2 RESULTS:", file = f)
        evaluate_model(level_2_y_pred, level_2_y_test, f, 2)
        print(file = f)
        print("Combined RESULTS:", file = f)
        evaluate_model(final_predictions, y_test, f, 3)
        print(file = f)
    f.close()

def evaluate_model_combined(combined_level_1_y_pred, combined_level_2_y_pred, combined_final_predictions, level_1_y_test, level_2_y_test, y_test, f):
    level_1_acc_array = []
    level_2_acc_array = []
    precision_array = []
    recall_array = []
    f1_array = []
    sub_acc_array = []
    for i in range(len(combined_final_predictions)):
        y_pred = combined_level_1_y_pred[i]
        level_1_acc = accuracy(level_1_y_test, y_pred)

        y_pred = combined_level_2_y_pred[i]
        level_2_acc = accuracy(level_2_y_test, y_pred)
        
        y_pred = combined_final_predictions[i]
        precision = Precision(y_test, y_pred)
        recall = Recall(y_test, y_pred)
        f1 = F1(precision, recall)
        sub_acc = SubsetAccuracy(y_test, y_pred)

        level_1_acc_array.append(level_1_acc)
        level_2_acc_array.append(level_2_acc)
        precision_array.append(precision)
        recall_array.append(recall)
        f1_array.append(f1)
        sub_acc_array.append(sub_acc)

    level_1_acc_array = np.array(level_1_acc_array)
    level_2_acc_array = np.array(level_2_acc_array)
    precision_array = np.array(precision_array)
    recall_array = np.array(recall_array)
    f1_array = np.array(f1_array)
    sub_acc_array = np.array(sub_acc_array)

    avg_level_1_acc = np.mean(level_1_acc_array)
    avg_level_2_acc = np.mean(level_2_acc_array)
    avg_precision = np.mean(precision_array)
    avg_recall = np.mean(recall_array)
    avg_f1 = np.mean(f1_array)
    avg_sub_acc = np.mean(sub_acc_array)

    std_level_1_acc = np.std(level_1_acc_array)
    std_level_2_acc = np.std(level_2_acc_array)
    std_precision = np.std(precision_array)
    std_recall = np.std(recall_array)
    std_f1 = np.std(f1_array)
    std_sub_acc = np.std(sub_acc_array)

    print('Level 1 accuracy average: ', avg_level_1_acc, file = f)
    print('Level 2 accuracy average: ', avg_level_2_acc, file = f)
    print('Precision average: ', avg_precision, file = f)
    print('Recall average: ', avg_recall, file = f)
    print('F1 average: ', avg_f1, file = f)
    print('Subset Accuracy average: ', avg_sub_acc, file = f)

    print('Level 1 accuracy std: ', std_level_1_acc, file = f)
    print('Level 2 accuracy std: ', std_level_2_acc, file = f)
    print('Precision std: ', std_precision, file = f)
    print('Recall std: ', std_recall, file = f)
    print('F1 std: ', std_f1, file = f)
    print('Subset Accuracy std: ', std_sub_acc, file = f)

def accuracy(y_test, y_pred):
    total = 0
    for i in range(len(y_pred)):
        if sum(y_test[i]) == 0:
            if sum(y_pred[i]) == 0:
                total += 1
            continue
        else:
            total += sum(np.logical_and(y_test[i], y_pred[i]))
    return total/len(y_test)

# Evaluates model and saves results to file
def evaluate_model(y_pred, y_test, f, level):

    # Calculate scores
    if level == 1 or level == 2:
        total = 0
        for i in range(len(y_pred)):
            if sum(y_test[i]) == 0:
                if sum(y_pred[i]) == 0:
                    total += 1
                continue
            else:
                total += sum(np.logical_and(y_test[i], y_pred[i]))
        print('Accuracy: ', total/len(y_pred), file = f)

    elif level == 3:
        precision = Precision(y_test, y_pred)
        recall = Recall(y_test, y_pred)
        print('Precision: ', precision, file = f)
        print('Recall: ', recall, file = f)
        print('F1: ', F1(precision, recall), file = f)
        print('Subset Accuracy: ', SubsetAccuracy(y_test, y_pred), file = f)

def Precision(y_true, y_pred):
    total = 0
    for i in range(0, len(y_pred)):
        total += sum(np.logical_and(y_true[i], y_pred[i]))/sum(y_pred[i])
    return total/len(y_true)

def Recall(y_true, y_pred):
    total = 0
    for i in range(0, len(y_pred)):
        total += sum(np.logical_and(y_true[i], y_pred[i]))/sum(y_true[i])
    return total/len(y_true)

def SubsetAccuracy(y_true, y_pred):
    return np.all(y_pred == y_true, axis = 1).mean()

def F1(precision, recall):
    f1 = (2 * precision * recall)/(precision+recall)
    return f1