import numpy as np
import pandas as p
import random
import math
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


if __name__ == '__main__':
    c = classify_nn(TRAINING_FILE, TESTING_FILE, 3)
    print(c)