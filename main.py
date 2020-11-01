from nn import nn
import numpy as np
import pandas as pd
import logging
import time

#Variables
timestr = time.strftime("%d-%m-%Y") 
levelLOG = logging.DEBUG

#Configuracion de logging
logging.basicConfig(filename = timestr+'.log',format='Time: %(asctime)s \tLevel: %(levelname)s \tFunc: %(funcName)s \tMsg: %(message)s', level=levelLOG)

# dataset1 = pd.read_csv('DataSet\\fashion-1.csv')
# print(dataset1.head())
logging.info("working")
NN = nn()
NN.create_empty_red()

print(NN.red[1][1].prev_Weights)