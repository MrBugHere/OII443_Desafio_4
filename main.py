from nn import nn
import numpy as np
import pandas as pd
import logging
import time

#Variables
timestr = time.strftime("%d-%m-%Y") 
levelLOG = logging.DEBUG

#HiperVariables
learning_rate = 0.5

#---------------------------------------------#

#Configuracion de logging
logging.basicConfig(filename = timestr+'.log',format='Time: %(asctime)s \tLevel: %(levelname)s \tFunc: %(funcName)s \tMsg: %(message)s', level=levelLOG)

#---------------------------------------------#


dataset1 = pd.read_csv('DataSet\\fashion-1.csv')
data = np.delete(pd.DataFrame.to_numpy(dataset1),0,1)

unique_labels = np.sort(dataset1.label.unique())
msk = np.random.rand(len(dataset1))< 0.7
train_data = dataset1[msk]
train_labels = np.transpose(pd.DataFrame.to_numpy(train_data.take([0], axis=1)))[0]
test_data   =   dataset1[~msk]
test_labels =   np.transpose(pd.DataFrame.to_numpy(test_data.take([0], axis=1)))[0]
train_data  =   np.delete(pd.DataFrame.to_numpy(train_data),0,1)
test_data   =   np.delete(pd.DataFrame.to_numpy(test_data),0,1)


#---------------------------------------------#
def softMax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#---------------------------------------------#

no_iter= 300

try:
    NN = nn()
    NN.create_empty_red([1784,20,len(unique_labels)])
    NN.train(train_data, train_labels, no_iter, learning_rate)
    NN.Save_State()
except:
    logging.exception("Se obtuvo Excepcion")
    raise