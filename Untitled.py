#!/usr/bin/env python
# coding: utf-8

# In[1]:

from typing import List

Tensor = List

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

x = shape([1, 2, 3])
print(x)  # Output: [3]

x = shape([[1, 2], [3, 4], [5, 6], [7, 8]])
print(x)  # Output: [4, 2]


# In[2]:

from typing import List

Tensor = List

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        if len(tensor) > 0 and isinstance(tensor[0], list):
            tensor = tensor[0] 
        return sizes

x = shape([1, 2, 3])
print(x) 

x = shape([[1, 2], [3, 4], [5, 6], [7, 8]])
print(x) 

# In[15]:

import numpy as np

def step_function(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def perceptron_output(weights: np.ndarray, bias: float, x: np.ndarray) -> float:
    calculation = np.dot(weights, x) + bias
    return step_function(calculation)

weights = np.array([2, 2])
bias = -3

print(perceptron_output(weights, bias, np.array([1, 1])))  # Output: 1.0
print(perceptron_output(weights, bias, np.array([0, 1])))  # Output: 0.0
print(perceptron_output(weights, bias, np.array([1, 0])))  # Output: 0.0
print(perceptron_output(weights, bias, np.array([0, 0])))  # Output: 0.0



# In[16]:


import numpy as np

def step_function(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def perceptron_output(weights, bias, x) -> float:
    calculation = np.dot(weights, x) + bias
    return step_function(calculation)
weights = np.array([2, 2])  
bias = -3

print(perceptron_output(weights, bias, [1, 1]))  
print(perceptron_output(weights, bias, [0, 1])) 
print(perceptron_output(weights, bias, [1, 0]))  
print(perceptron_output(weights, bias, [0, 0]))  


# In[ ]:





# In[ ]:


import math
from typing import List

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot(weights: List[float], inputs: List[float]) -> float:
    return sum(w * i for w, i in zip(weights, inputs))

def neuron_output(weights: List[float], inputs: List[float]) -> float:
    return sigmoid(dot(weights, inputs))

def feedforward(neural_network: List[List[List[float]]], input_vector: List[float]) -> List[float]:
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return output

XOR_network = [
    # Hidden layer
    [[20, 20, -30], [20, 20, -10]],
    # Output layer
    [[-10, 60, -30]]
]

for x in [0, 1]:
    for y in [0, 1]:
        print(x, y, feedforward(XOR_network, [x, y]))




# In[ ]:

import numpy as np

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feedforward(neural_network, input_vector):
    output = input_vector
    for layer in neural_network:
        input_with_bias = output + [1] 
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
    return output

XOR_network = [
    [[20, 20, -30], [20, 20, -10]],  
    [[-10, 60, -30]]  
]

for x in [0, 1]:
    for y in [0, 1]:
        input_vector = [x, y]  
        output = feedforward(XOR_network, input_vector)  
        print(f"Input: ({x}, {y}) -> Output: {output[-1]:.4f}")





