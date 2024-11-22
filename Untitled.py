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














import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Standardsealer, LabelEncoder

data={
    'gender':['male','female','female','male','female','male','female','male','female','male']
    'study_time':[10,8,12,15,7,1,6,10,9,11],
    'math_score':[88,92,78,85,94,75,70,88,90,77],
    'reading_score':[93,89,80,84,88,85,85,76,89,91,82],
    'writing_score':[84,90,75,80,85,78,1,86,88,76],
    'passed_exam':[1,1,0,1,1,0,0,1,1,0];    
}
df=pd.DataFrame(data)
np.random.seed(42)
nan_indices=np.random.choice(df.index,size=2,replace=False)
df.loc[nan_indices,'math_score']=np.nan
df_cleaned=df.dropna()
label_encoder=LabelEncoder()
df_cleaned['gender']=label_encoder.fit_transform(df_cleaned['gender'])














# In[ ]:




