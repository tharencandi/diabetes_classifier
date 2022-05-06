import string
import numpy as np
import pandas as p
import random
import math
from scipy.stats import norm

TRAINING_FILE = "./diabetes_CFS.csv"
TESTING_FILE = "./test.csv"
NUM_ATTRIBUTES = 5

"""
Shuffle the dataset randomly.
Split the dataset into k groups
For each unique group:
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
Summarize the skill of the model using the sample of model evaluation scores
"""
def generate_k_folds(filename, k):

    training_df = p.read_csv(filename, header=None )
    training_df[NUM_ATTRIBUTES] = training_df[NUM_ATTRIBUTES].map({'yes': 1, 'no': 0})

    #seperating classes to numpy arrays and randomise order
    yes_class = np.array(training_df[training_df[NUM_ATTRIBUTES] == 1].values)
    no_class =  np.array(training_df[training_df[NUM_ATTRIBUTES] == 0].values)
    np.random.shuffle(yes_class)
    np.random.shuffle(no_class)

    yes_rows = yes_class.shape[0]
    no_rows = no_class.shape[0]
    folds = []
    for i in range(1,k+1):
        fold = []
        for y in range((i-1)*(yes_rows // k),i *(yes_rows // k)):
            fold.append(yes_class[y])
        for n in range((i-1)*(no_rows // k),i *(no_rows // k)):
            fold.append(no_class[n])
        folds.append(fold)

    #remainder
    for y in range(k * (yes_rows//k), yes_rows):
        folds[y % len(folds)].append(yes_class[y])
    for n in range(k * (no_rows // k), no_rows):
        folds[n % len(folds)].append(no_class[n])
    
 
    
    nb_sum = 0
    nn_sum = 0
    for b in range(len(folds)):
        testing_str = format_fold(folds[b])
        training_str = ""
       
        for i in range(len(folds)):
            if i == b:
                continue
            training_str += format_fold(folds[i])
        f_train = open("fold_training.csv", "w")
        f_test= open("fold_testing.csv", "w")
        f_train.write(training_str)
        f_test.write(testing_str)
        f_test.close()
        f_train.close()
        nb_r = classify_nb("fold_training.csv","fold_testing.csv")
        nn_r = classify_nn("fold_training.csv","fold_testing.csv", 5)
 
        nn_correct = 0
        nb_correct = 0
        testing_fold = folds[b]
     
        for i in range(len(testing_fold)):
            if (nb_r[i] == "yes" and testing_fold[i][NUM_ATTRIBUTES] == 1) or (nb_r[i] == "no" and testing_fold[i][NUM_ATTRIBUTES] == 0):
                nb_correct += 1
            if (nn_r[i] == "yes" and testing_fold[i][NUM_ATTRIBUTES] == 1) or (nn_r[i] == "no" and testing_fold[i][NUM_ATTRIBUTES] == 0):
                nn_correct += 1

        nb_sum += (nb_correct / len(testing_fold))
        nn_sum += (nn_correct / len(testing_fold))

    print("NB accuracy:", nb_sum/k)
    print("NN accuracy:", nn_sum/ k)
    
    

def format_fold(fold):
    s = ""
    for row in fold:
        for j in range(len(row)):
            if j == NUM_ATTRIBUTES:
                if row[j] == 0:
                    s += "no"
                else:
                    s+= "yes"
            else:
                s += str(row[j])+","
        s += "\n"
    return s
# 7 variables
def distance(A, B):
    result = 0
    ssum = 0
    for i in range(NUM_ATTRIBUTES):
        ssum += (A[i] - B[i])*(A[i] - B[i])
    result = math.sqrt(ssum)
    return result

"""
K nearest neighbour algorithm w/
Euclidiean distance measurement
"""
def classify_nn(training_filename, testing_filename, k):

    classes = []
    
    testing = np.genfromtxt(testing_filename, delimiter=',')
    
    training_df = p.read_csv(training_filename, header=None )
    training_df[NUM_ATTRIBUTES] = training_df[NUM_ATTRIBUTES].map({'yes': 1, 'no': 0})
    training = np.array(training_df.values)
    #training = training_df.to_numpy()
    
    for to_class in testing:
        D = []
        chosen_class = "no"
        for t_data in training:
            D.append(distance(to_class, t_data))

        d_indicies = np.argsort(D)   
        d_indicies = d_indicies[0:k]
        
        
        num_ones = 0
        for index in d_indicies:
            num_ones += training[index,NUM_ATTRIBUTES]
        num_zeros = k - num_ones

        if num_ones >= num_zeros:
            chosen_class = "yes"

        
        classes.append(chosen_class)

    return classes

def classify_nb(training_filename, testing_filename):
    testing = np.genfromtxt(testing_filename, delimiter=',')
    
    training_df = p.read_csv(training_filename, header=None )
    training_df[NUM_ATTRIBUTES] = training_df[NUM_ATTRIBUTES].map({'yes': 1, 'no': 0})
    
    yes_class = np.array(training_df[training_df[NUM_ATTRIBUTES] == 1].values)
    no_class = np.array(training_df[training_df[NUM_ATTRIBUTES] == 0].values)
  
    classes = []
    yes_prob = len(yes_class) / len(training_df) 
    no_prob = len(no_class)/ len(training_df)
    
  
    yes_means = []
    no_means = []
    yes_stds = []
    no_stds = []

    for col in range(yes_class.shape[1]):
        x = yes_class[:,col]
        mean = np.mean(x)
        std = np.std(x)

        yes_means.append(mean)
        yes_stds.append(std)

    for col in range(no_class.shape[1]):
        x = no_class[:,col]
        mean = np.mean(x)
        std = np.std(x)
        no_means.append(mean)
        no_stds.append(std)

    for row in testing:
        c = "no"
        yes = yes_prob
        no = no_prob
        for i in range(NUM_ATTRIBUTES):
            y_mean = yes_means[i]
            y_std = yes_stds[i]
            y_a = row[i]
            res = norm.pdf(y_a, y_mean, y_std)
            yes *= res

            n_mean = no_means[i]
            n_std = no_stds[i]
            n_a = row[i]
            res = norm.pdf(n_a, n_mean, n_std)
            no *= res
        
        if yes >= no:
            c = "yes"
        classes.append(c)

    return classes

if __name__ == '__main__':
    #c = classify_nn(TRAINING_FILE, TESTING_FILE, 5)
    #b = classify_nb(TRAINING_FILE, TESTING_FILE)
    #print("knn: ", c)
    #print("bayes: ", b)
    generate_k_folds(TRAINING_FILE, 10)
