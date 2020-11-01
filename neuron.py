import math
import logging

#HiperVariables
b = 0

class neuron:
    
    def __init__(self, prev_layer, prev_weights, next_layer, value):
        self.prev_Layer = prev_layer
        self.prev_Weights =  prev_weights
        self.aux_Weights = []
        self.next_Layer = next_layer
        self.value = value     
    

    def activation_function(self):
        return_value = 1/( 1 + math.exp(-self.value))
        return return_value
 
    def deriv_f(self, ):
        return_value = self.activation_function()*(1 - self.activation_function())
        return return_value

    def calculate_valor(self):
        if self.prev_Layer is None:
            logging.warning("CAPA DE ENTRADA, NO SE PUEDE CALCULAR VALORES A ESTA")
        else:
            ret = 0
            for i in range(len(self.prev_Layer)):
                logging.info("largo weights: %s largo capa anterior: %s ",len(self.prev_Weights), len(self.prev_Layer))
                ret += self.prev_Weights[i] * self.prev_Layer[i].activation_function()
            return ret + b

    
