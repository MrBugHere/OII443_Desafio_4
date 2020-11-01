import math
import logging

# HiperVariables
b = 0


class neuron:

    def __init__(self, prev_layer, prev_weights, next_layer, value, pos):
        self.prev_Layer = prev_layer
        self.prev_Weights = prev_weights
        self.aux_Weights = []
        self.next_weights = []
        self.next_Layer = next_layer
        self.value = value
        self.delta = None
        self.gradient = None
        self.pos = pos

    def activation_function(self):
        return_value = 1 / (1 + math.exp(-self.value))
        return return_value

    def deriv_f(self):
        return_value = self.activation_function() * (1 - self.activation_function())
        return return_value

    def calculate_valor(self):
        if self.prev_Layer is None:
            logging.warning("CAPA DE ENTRADA, NO SE PUEDE CALCULAR VALORES A ESTA")
        else:
            for i in range(len(self.prev_Layer)):
                ret += self.prev_Weights[i] * self.prev_Layer[i].activation_function()
            return ret + b

    def calculate_new_weights(self, learning_rate, expected_value):
        if self.next_Layer is None:
            self.delta = (self.activation_function() - expected_value) * self.deriv_f()
        else:
            summation = 0
            for n in self.next_Layer:
                summation += n.delta * n.prev_Weights[self.pos]
            self.delta = summation * self.deriv_f()

            for n in self.next_Layer:
                new_weight = -learning_rate * n.delta * self.activation_function()
                self.aux_Weights.append(new_weight)

    #al final partes de la primera capa y se hace self.next_weights=self.aux_Weights lo mismo para todas las otras capas
    #solo que tambien actualizas previous weights en el proceso (usas un for de n en prev_layer y haces
    #self.prev_weight[i] = n.next_weight[self.pos] y luego i+=1

    def update_weights(self):
        self.next_weights = self.aux_Weights
        i = 0
        for n in self.prev_Layer:
            self.prev_Weights[i] = n.next_weights[self.pos]
            i += 1
        #una version basica de lo que comente arriba, queda checkear la capa inicial y final y por ultimo en nn llamar a
        # esta funcion para cada neurona partiendo desde la primera capa
