import math
class neuron:
    
    def __init__(self, prev_layer, prev_weights, next_layer, value):
        self.prev_Layer = prev_layer
        self.prev_Weights =  prev_weights
        self.next_Layer = next_layer
        self.value = value      
    

    def activation_function(self, w):
        return_value = 1/( 1 + math.exp(-w))
        return return_value
 
    def deriv_f(self, w):
        return_value = self.activation_function(w)*(1 - self.activation_function(w))
        return return_value

    def calculate_valor(self):
        for w in self.prev_Weights:
            ret += w * self.activation_function(w)
        return w 

    
