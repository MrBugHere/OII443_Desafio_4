from nn import nn
import numpy as np
import pandas as pd
import logging
import time

#Variables
timestr = time.strftime("%d-%m-%Y") 
levelLOG = logging.DEBUG
#HiperVariables

epoch = 2
train_percent = 0.7
#---------------------------------------------#

#Configuracion de logging
logging.basicConfig(filename = timestr+'.log',format='Time: %(asctime)s \tLevel: %(levelname)s \tFunc: %(funcName)s \tMsg: %(message)s', level=levelLOG)

#---------------------------------------------#


dataset1 = pd.read_csv('DataSet\\fashion-1.csv')
data = np.delete(pd.DataFrame.to_numpy(dataset1),0,1)
labels = np.transpose(pd.DataFrame.to_numpy(dataset1.take([0], axis=1)))[0]
unique_labels = np.sort(dataset1.label.unique())

#---------------------------------------------#
def softMax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def train(data_Set, Expected_R, epoch=1):
    f= open(timestr+"_output.txt","w+")
    NN = nn()
    iterations = int(len(data_Set)*train_percent)
    i25 =  int(iterations*0.25)
    i50 =  int(iterations*0.5)
    i75 =  int(iterations*0.75)
    NN.create_empty_red([1784,20,len(unique_labels)])
    for _ in range(epoch):
        for i in range(iterations):
            f.write("Inicio de Epoch")
            ret = NN.predict(data_Set[i])
            if(i == i25):
                f.write("porcentaje 25 %d - %d - %d",i,ret,Expected_R[i])
            if(i == i50):
                f.write("porcentaje 50 %d - %d - %d",i,ret,Expected_R[i])
            if(i == i75):
                f.write("porcentaje 75 %d - %d - %d",i,ret,Expected_R[i])
            NN.update_weights(unique_labels)
            logging.info("Actualizacion de Pesos")
    NN.Save_State()
#---------------------------------------------#

try:
    train(data, labels, epoch)
    # print(data[1])

    NN = nn()
    NN.create_empty_red([1784,20,len(unique_labels)])
    ret = NN.predict(data[1])
    print(ret)
    # # NN.Save_State()
    # # NN.Load_State("Save_State.pickle")

    # print(len(NN.red[0]))

except:
    logging.exception("Se obtuvo Excepcion")
    raise