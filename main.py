from nn import nn
import numpy as np
import pandas as pd
import logging
import time

#Variables
timestr = time.strftime("%d-%m-%Y") 
levelLOG = logging.DEBUG
#---------------------------------------------#

#Configuracion de logging
logging.basicConfig(filename = timestr+'.log',format='Time: %(asctime)s \tLevel: %(levelname)s \tFunc: %(funcName)s \tMsg: %(message)s', level=levelLOG)
#---------------------------------------------#


dataset1 = pd.read_csv('DataSet\\fashion-1.csv')
data = np.delete(pd.DataFrame.to_numpy(dataset1),0,1)
labels = np.transpose(pd.DataFrame.to_numpy(dataset1.take([0], axis=1)))[0]
unique_labels = np.sort(dataset1.label.unique())

NN = nn()
NN.create_empty_red([1784,20,len(unique_labels)])
logging.info("Arreglo final: %s", NN.predict(data[0]))