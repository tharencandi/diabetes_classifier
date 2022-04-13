import numpy as np
FILE = "./diabetes_norm.csv"

my_data = np.genfromtxt(FILE, delimiter=',')

print(my_data.shape)
print(my_data)