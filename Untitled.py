#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

# Define input and output arrays
x = np.array([[2,9], [1,9], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Normalize the data
x = x / np.amax(x, axis=0)
y = y / 100

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivation_sigmoid(x):
    return x * (1 - x)

# Set parameters
epoch = 5000
lr = 0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
outputlayer_neurons = 1

# Initialize weights and biases
wb = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bb = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, outputlayer_neurons))
bout = np.random.uniform(size=(1, outputlayer_neurons))

# Training the neural network
for i in range(epoch):
    # Forward propagation
    hinp1 = np.dot(x, wb)
    hinp = hinp1 + bb
    hlayer_act = sigmoid(hinp)
    
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    
    # Backpropagation
    EO = y - output
    outgrad = derivation_sigmoid(output)
    d_output = EO * outgrad
    
    EH = d_output.dot(wout.T)
    hiddengrad = derivation_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    
    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wb += x.T.dot(d_hiddenlayer) * lr

# Print results
print("INPUT:\n", x)
print("ACTUAL:\n", y)
print("PREDICTED:\n", output)
















# In[ ]:




