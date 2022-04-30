import numpy as np
import pandas as p
import random
import math
from scipy.stats import norm

TRAINING_FILE = "./diabetes_norm.csv"
TESTING_FILE = "./test.csv"


# 7 variables
def distance(A, B):
    result = 0
    ssum = 0
    for i in range(8):
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
    
    training_df[8] = training_df[8].map({'yes': 1, 'no': 0})
    training = training_df.to_numpy()
    
    for to_class in testing:
        D = []
        chosen_class = "no"
        for t_data in training:
            D.append(distance(to_class, t_data))

        d_indicies = np.argsort(D)   
        d_indicies = d_indicies[0:k]
        
        num_ones = 0
        for index in d_indicies:
            num_ones += training[index,8]
        num_zeros = k - num_ones

        """if num_ones == num_zeros:
            chosen_class = random.choice(["yes", "no"])"""
        if num_ones >= num_zeros:
            chosen_class = "yes"

        
        classes.append(chosen_class)
    return classes



def classify_nb(training_filename, testing_filename):
    testing = np.genfromtxt(testing_filename, delimiter=',')
    
    training_df = p.read_csv(training_filename, header=None )
    training_df[8] = training_df[8].map({'yes': 1, 'no': 0})

    #seperating classes to numpy arrays
    yes_class = training_df[training_df[8] == 1].to_numpy()
    no_class = training_df[training_df[8] == 0].to_numpy()
  
    classes = []

    #basic probability of any yes or any no
    yes_prob = len(yes_class) / len(training_df) 
    no_prob = len(no_class)/ len(training_df)
    
    # we need this information to calculate the probability of P(ai | yes) and P(ai | no) from the normal curve
    yes_means = []
    no_means = []
    yes_stds = []
    no_stds = []

    #get means and stds for all attributes for no and yes classes
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


    #actually calculate probability of row in input data to be in class yes and no
    #determine its output class
    for row in testing:
        c = "no"
        yes = yes_prob
        no = no_prob
        for i in range(8):
            y_mean = yes_means[i]
            y_std = yes_stds[i]
            y_a = row[i]
            res = norm.cdf(y_a, y_mean, y_std)
            yes *= res

            n_mean = no_means[i]
            n_std = no_stds[i]
            n_a = row[i]
            res = norm.cdf(n_a, n_mean, n_std)
            no *= res
        print(yes, no)
        if yes >= no:
            c = "yes"
        classes.append(c)

    return classes
        
        
    
   
    
    




    return []

if __name__ == '__main__':
    c = classify_nn(TRAINING_FILE, TESTING_FILE, 5)
    b = classify_nb(TRAINING_FILE, TESTING_FILE)
    print("knn: ", c)
    print("bayes: ", b)