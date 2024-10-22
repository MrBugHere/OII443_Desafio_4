import random
from neuron import neuron
import numpy as np
import logging
# import pickle
import joblib

class d:
    def __init__(self, arrayIn, no_iter, total_it, percent):
        self.Array = arrayIn
        self.No_iter = no_iter
        self.Total_it = total_it
        self.Percent = percent

class nn:

    def __init__(self):
        self.red = []

    def connect_layers(self):
        for i in range(len(self.red)):
            for j in range(len(self.red[i])):
                if i == 0:
                    self.red[i][j].next_Layer = self.red[i+1]
                elif i < (len(self.red) - 1):
                    self.red[i][j].next_Layer = self.red[i+1]
                    self.red[i][j].prev_Layer = self.red[i-1]
                elif i == len(self.red) - 1:
                    self.red[i][j].prev_Layer = self.red[i-1]

    def create_empty_red(self, order=[1, 3, 1]):
        for i in range(len(order)):
            if i == 0:  # Capa de entrada
                layer = []
                for z in range(order[i]):  # se crea la cantidad de neuronas en la capa
                    layer.append(neuron(None, None, None, 0, z))
                layer.append(neuron(None, None, None, None, None)) #se le añade el bias neuron
                self.red.append(layer)
                logging.info("Se creo capa de entrada con %s neuronas", order[i])

                continue

            if i < (len(order) - 1):  # Capa Oculta
                layer = []
                for z in range(order[i]):  # se crea la cantidad de neuronas en la capa
                    layer.append(neuron(None, np.random.uniform(low=0, high=1, size=(len(self.red[-1]))), None, 0, z))
                layer.append(neuron(None, len(self.red[-1]), None, None, None))  # se le añade el bias neuron
                self.red.append(layer)  # conecta la capa actual a NN
                logging.info("Se creo capa de oculta con %s neuronas", order[i])
                continue

            if i == len(order) - 1:  # Capa de Salida
                layer = []
                for z in range(order[i]):  # se crea la cantidad de neuronas en la capa
                    layer.append(neuron(None, np.random.uniform(low=0, high=1, size=(len(self.red[-1]))), None, 0, z))
                self.red.append(layer)  # conecta la capa actual a NN
                logging.info("Se creo capa de Salida con %s neuronas", order[i])
                continue
        self.connect_layers()

    def train(self, x, y, num_iter, learning_rate):
        i25 =  int(num_iter*0.25)
        i50 =  int(num_iter*0.5)
        i75 =  int(num_iter*0.75)
        iFinal = num_iter - 1
        save = []
        positions = []
        for i in range(num_iter):
            high = len(x)-1
            pos = int(high*random.random())
            while pos in positions:
                pos = int(high*random.random())
            positions.append(pos)
            input = x[pos]
            expected_output = y[pos]
            ret = self.predict(input)
            if i == 1:
                logging.info("Se inicia entrenamiento")
            if i == i25:
                logging.info("Se guardo dato del 25 por ciento completado")
                save.append(d(ret,i,num_iter,25))
            if i == i50:
                logging.info("Se guardo dato del 50 por ciento completado")
                save.append(d(ret,i,num_iter,50))
            if i == i75:
                logging.info("Se guardo dato del 75 por ciento completado")
                save.append(d(ret,i,num_iter,75))
            if i == iFinal:
                logging.info("Se guardo dato del 100 por ciento completado")
                save.append(d(ret,i,num_iter,100))
            self.backpropagate(learning_rate, ret, expected_output)
        fileName = 'Collected_Data.sav'
        joblib.dump(save,fileName)

    def backpropagate(self, learning_rate, output, expected_value):
        expected_vector = []
        for i in range(len(self.red[-1])):
            if expected_value == i:
                expected_vector.append(1)
            else:
                expected_vector.append(0)
        i = 0
        for neuron in self.red[-1]:
            neuron.calculate_new_weights(learning_rate, output[i], expected_vector[i])
            i+=1
        for i in range(2, len(self.red)+1):
            for neuron in self.red[-i]:
                neuron.calculate_new_weights(learning_rate)
        if not self.red[0][0].next_weights:
            self.set_next_weights()
        self.update_weights()

    def set_next_weights(self):
        for i in range(len(self.red)):
            for n in self.red[i]:
                n.set_next_weights()

    def update_weights(self):
        for i in range(len(self.red)):
            for neuron in self.red[i]:
                neuron.update_weights()

    def predict(self, input):
        for i in range(len(input)):
            self.red[0][i].value = input[i]

        for i in range(1, len(self.red)):
            for j in range(len(self.red[i])):
                self.red[i][j].update_valor()
        ret = []
        for i in range(len(self.red[-1])):
            ret.append(self.red[-1][i].activation_function())

        return ret

    def Save_State(self):
        logging.info("Guardando Estado")
        fileName = 'finalized_trained_data.sav'
        joblib.dump(self.red,fileName)
        logging.info("Estado Guardado")

    def Load_State(self):
        logging.info("Cargando Estado Previo")
        fileName = 'finalized_trained_data.sav'
        self.red = joblib.load(fileName)
