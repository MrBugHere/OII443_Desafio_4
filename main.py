from nn import nn
import numpy as np
import pandas as pd

# dataset1 = pd.read_csv('DataSet\\fashion-1.csv')
# print(dataset1.head())

NN = nn()
NN.create_empty_red()

print(NN.red[1][1].prev_Weights)