#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List
Tensor = List
def shape(tensor:Tensor)->List[int]:
    sizes:list[int]=[]
        while isinstance(tensor,list):
            sizes.append(len(tensor))
            tensor = tensor[0]
            return sizes
        x = shape([1,2,3])
        print(x)
        x = shape(([1,2],[3,4],[5,6],[7,8]))
        print(x)


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

def step_function(x:float)->float:
    return 1.0 if x >0 else 0.0

    def perception_output(weights,bias,x)->float:
        calculation = dot(weights,x)+bias
        return step_function(calculation)
    weights[2,2]
    bias=-3
    print(perception_output(weights,bias,[1,1]))
    print(perception_output(weights,bias,[0,1]))
    print(perception_output(weights,bias,[1,0]))
    print(perception_output(weights,bias,[0,0]))


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





# In[20]:


def neurox_output(weights,input):
    return sigmoid(dot(weights,inputs))

def feedforward(neural_network,input_vector):
    output=[]
    
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron,input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector=output
        return output
    
XOR_network = [# a hidden layer[[20,20,-30],[20,20,-10]],#output layer[[-10,60,-30]]
for x in[0,1]:
    for y in [0,1]:
        print x,y,feed_forward
        (XOR_network[x,y])[-1]


# In[22]:


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


# In[ ]:





# In[ ]:




