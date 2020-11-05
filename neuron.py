from math import exp
from random import random as random


class neuron:

    def __init__(self, prev_layer, prev_weights, next_layer, value, pos):
        self.prev_Layer = prev_layer
        if prev_weights is None:
            self.prev_Weights = prev_weights
        elif isinstance(prev_weights, int):
            bias = []
            for n in range(prev_weights): bias.append(random())
            self.prev_Weights = bias
        else:
            self.prev_Weights = prev_weights.tolist()
        self.aux_Weights = []
        self.next_weights = []
        self.next_Layer = next_layer
        self.value = value
        self.delta = None
        self.pos = pos

    def activation_function(self):
        if self.pos is None:  # caso de neurona bias
            return 1
        return 1 / (1 + exp(-self.value))

    def deriv_f(self):
        if self.pos is None:
            return 1
        return self.activation_function() * (1 - self.activation_function())

    def calculate_valor(self):
        if self.prev_Layer is None:
            return self.activation_function()
        else:
            ret = 0
            for i in range(len(self.prev_Layer)):
                ret += self.prev_Weights[i] * self.prev_Layer[i].activation_function()
            return ret

    def update_valor(self):
        self.value = self.calculate_valor()

    def calculate_new_weights(self, learning_rate, output=None, expected_value=None):
        if self.next_Layer is None:
            self.delta = (output - expected_value) * self.deriv_f()
        elif self.pos is None:
            summation = 0
            for n in self.next_Layer:
                if n.pos is None:
                    continue
                summation += n.delta * n.prev_Weights[-1]
            self.delta = summation * self.deriv_f()

            for n in self.next_Layer:
                new_weight = -learning_rate * n.delta * self.activation_function()
                self.aux_Weights.append(new_weight)
        else:
            summation = 0
            for n in self.next_Layer:
                if n.pos is None:
                    continue
                summation += n.delta * n.prev_Weights[self.pos]
            self.delta = summation * self.deriv_f()

            for n in self.next_Layer:
                new_weight = -learning_rate * n.delta * self.activation_function()
                self.aux_Weights.append(new_weight)

    def set_next_weights(self):
        if self.pos is None:
            self.next_weights = self.aux_Weights.copy()
            i = 0
            for n in self.next_Layer:
                self.next_weights[i] = n.prev_Weights[-1]
                i+=1
        elif self.next_Layer is None:
            pass
        else:
            self.next_weights = self.aux_Weights.copy()
            i = 0
            for n in self.next_Layer:
                self.next_weights[i] = n.prev_Weights[self.pos]
                i += 1

    def update_weights(self):
        if self.prev_Layer is None:
            self.next_weights += self.aux_Weights
        elif self.next_Layer is None:
            i = 0
            for n in self.prev_Layer:
                self.prev_Weights[i] = n.next_weights[self.pos]
                i += 1
        elif self.pos is None:
            self.next_weights += self.aux_Weights
        else:
            i = 0
            self.next_weights += self.aux_Weights
            for n in self.prev_Layer:
                self.prev_Weights[i] = n.next_weights[self.pos]
                i += 1
