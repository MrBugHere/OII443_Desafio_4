import math 
from neuron import neuron
import numpy as np
import logging

#HiperVariables

learning_rate = .5


class nn:
    def __init__(self):
        self.red = []

    def create_empty_layer(self, type_layer, prev_layer, prev_weights, no_neurons):
        
        if type_layer == 1:#Capa de entrada
            layer = []
            for _ in range(no_neurons):#se crea la cantidad de neuronas en la capa
                layer.append(neuron(prev_layer,None,None, 0))
            
            self.red.append(layer)
            
        elif type_layer == 2:#Capa Oculta
            layer = []
            for _ in range(no_neurons):#se crea la cantidad de neuronas en la capa
                layer.append(neuron(prev_layer, np.random.uniform(low=0, high=1, size=(prev_weights)), None, 0))
            
            for n in self.red[-1]: #conecta cada neurona de la capa anterior a la actual
                n.next_layer = layer            
            self.red.append(layer) #conecta la capa actual a NN
                

        elif type_layer == 3: #capa salida
            layer = []
            for _ in range(no_neurons): #se crea la cantidad de neuronas en la capa
                layer.append(neuron(prev_layer, np.random.uniform(low=0, high=1, size=(prev_weights)), None, 0))
            for n in self.red[-1]:
                n.next_layer = layer #conecta cada neurona de la capa anterior a la actual            
            
            self.red.append(layer)  #conecta la capa actual a NN
        else:
           logging.warning("Se intento de crear una capa inexistente")


    def create_empty_red(self, order = [1,3,1]):
        for i in range(len(order)):
            if i == 0: #Capa de entrada
                self.create_empty_layer(1, None, None, order[i])
                logging.info("Se creo capa de entrada con %s neuronas",  order[i])
                continue           

            if i < (len(order)-1): #Capa Oculta
                self.create_empty_layer(2, self.red[i-1], order[i-1], order[i])
                logging.info("Se creo capa de oculta con %s neuronas",  order[i])
                continue

            if i == len(order)-1 : #Capa de Salida
                self.create_empty_layer(3, self.red[i-1], order[i-1], order[i])
                logging.info("Se creo capa de Salida con %s neuronas",  order[i])
                continue

    def predict(self, input):
        for i in range(len(input)):
            self.red[0][i].value = input[i]
        
        for i in range(1, len(self.red)):
            for j in range(len(self.red[i])):
                self.red[i][j].value = self.red[i][j].calculate_valor()
        ret = []
        for i in range(len(self.red[-1])):
            ret.append(self.red[-1][i].value)

        return ret
