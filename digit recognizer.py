#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Install a conda package in the current Jupyter kernel
# import sys
# !conda install --yes --prefix {sys.prefix} tensorflow


# In[2]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


import os


# In[4]:


# importing the libraries required for neural networks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[5]:


# getting the training data
train_data = pd.read_csv('train.csv')


# In[6]:


train_data.head()


# In[7]:


# splitting the input features and target features
X = train_data.drop(['label'],axis =1)
y = train_data['label'].values


# In[8]:


# Tensorflow layers expects the inputs in the form of Array
# Conv2D expects the dimension to be ndim=4
X = np.array(X)
np.shape(X)


# In[9]:


X = X.reshape(42000,28,28,1)


# In[10]:


np.shape(X)


# In[11]:


# splitting the data into training and cross validation
# holdout validation method
X_train, X_holdout, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[12]:


# Define the number of classes
classes = len(set(y))


# In[13]:


# Create a neural network with 13 deep hidden layers + 1 input and 1 output layer
cnn_model = Sequential (name ='C-Neural_Network_Model')
cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu', input_shape = (28,28,1), name = 'First_layer'))


# In[14]:


# cnn_model.add(MaxPooling2D(pool_size = (2,2), name = 'Second_layer'))


# In[15]:


# cnn_model.add(Dropout(rate = 0.2, name = 'Third_layer'))


# In[16]:


# cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', name = 'Fourth_layer'))
# cnn_model.add(MaxPooling2D(pool_size = (2,2), name = 'Fifth_layer'))


# In[17]:


# cnn_model.add(Dropout(rate = 0.2, name = 'Sixth_layer'))


# In[18]:


# cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', name = 'Seventh_layer'))


# In[19]:


# cnn_model.add(MaxPooling2D(pool_size = (2,2),name = 'Eighth_layer'))


# In[20]:


# cnn_model.add(Dropout(rate = 0.15, name = 'nineth_layer'))


# In[21]:


# cnn_model.add(Flatten(name = 'tenth_layer'))


# In[22]:


cnn_model.add(Dense(units = 128, activation = 'relu', name = 'eleventh_layer'))


# In[23]:


# cnn_model.add(Dropout(rate = 0.1, name = 'twelfth_layer'))


# In[24]:


# cnn_model.add(Dense(units = 32, activation = 'relu', name = 'Thirteenth_layer'))


# In[25]:


# cnn_model.add(Dropout(rate = 0.1, name = 'fourteenth_layer'))


# In[26]:


cnn_model.add(Dense(units = classes, activation = 'softmax', name = 'Output_layer'))


# In[27]:


cnn_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[28]:


fitted_model = cnn_model.fit(X_train, y_train, epochs = 1)


# In[ ]:


# plotting the accuracy for each epoch
accuracy = fitted_model.history['acc']
plt.plot(range(len(accuracy)), accuracy, 'o', label = 'accuracy')
plt.title('Accuracy of the model for each epoch')
plt.legend()


# In[ ]:


evaluation = cnn_model.predict_classes(X_holdout)


# In[ ]:


print(classification_report(evaluation, y_test))


# In[ ]:


test_data = pd.read_csv('test.csv')


# In[ ]:


test_data = np.array(test_data)
test_data = test_data.reshape(28000,28,28,1)


# In[ ]:


predictions = cnn_model.predict_classes(test_data)


# In[ ]:


submissions = pd.DataFrame({'ImageId':range(1, len(test_data)+1), 'Label':predictions})


# In[ ]:


submissions.to_csv('submissions.csv', index = False)


# In[ ]:




