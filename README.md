## OII443_Desafio_3

### Descripción
Juan está interesado en el mercado de la venta de ropa en línea. Su revolucionaria idea implica usar una inteligencia artificial que recomiende ropa que se parezca. Para poder realizar esto, Juan deberá primero clasificar los tipos de ropa. Para ello, se les ha pedido a los alumnos que implementen una inteligencia artificial que haga el trabajo.

### Solución
Para identificar ropa similar se utilizara una red neuronal, la cual contiene las clases neuron y nn:

```python
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
``` 

```python
class nn:

    def __init__(self):
        self.red = []

``` 

En donde la clase nn, contiene todas las neuronas y sus conexiones. Ademas tiene las funciones create_layer(), create_empty_layer(), create_empty_red(); las cuales se utilizan para crear la red neuronal. Luego en la clase neuron, tiene las funciones de activation_function(), deriv_f(), calculate_valor(), calculate_new_weights(); funciones que permiten el funcionamiento de la red neuronal, actualizando pesos o retornando la funcion de activacion de cada neurona.

### Ejecucion

En primer lugar se procesa el dataset utilizando la libreria numpy, el dataset puede encontrarse en "https://github.com/OptativoPUCV/Fashion-DataSet". Donde unique_labels son los tipos de ropa a los que se pretende que la red neuronal clasifique de manera similar, esta variable contiene los números desde el 0 al 9. Luego se crea la red neuronal vacía y posteriormente se empieza poblar la red con los pesos correspondientes, para luego comenzar a entrenar la red neuronal.