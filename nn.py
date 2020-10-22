import math 
from neuron import neuron
import numpy as np

class nn:
    def __init__(self):
        self.red = []

    def create_layer(self, type_layer, prev_layer, prev_weights, no_neurons):
        
        if type_layer == 1:#Capa de entrada
            layer = []
            for i in range(0, no_neurons-1):
            
        elif type_layer == 2:#Capa Oculta
             print('ERROR tipo de capa no existente')
        elif type_layer == 3:
             print('ERROR tipo de capa no existente')
        else:
            print('ERROR tipo de capa no existente')


    def create_empty_red(self, order = [1,3,1]):
        for i in len(order):
            if i == 0: #Capa de entrada
                create_layer(1, None, None, order[i])
                continue
            
            if i <= len(order) - 1: #Capa Oculta
                create_layer(2, red[i-1], np.random.rand(order[i]), order[i])
                continue

            if i == len(order): #Capa de Salida

                continue