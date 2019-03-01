
# coding: utf-8

# In[ ]:


import numpy as np
import os
import cv2
import keras
import sklearn
import pandas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
from keras import Model
from keras.callbacks import TensorBoard


# In[ ]:


labels=['violent','non_violent']

# path = "C:/Users/NBH/Desktop/raw/dataset/train/10_FPS/30x30/"
# for label in labels:
#     vector=[]

#     images = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
#     for image in images:
#         img=cv2.imread(image,0)
#         #print(img)
#         vector.append(img)
#     vector = np.asarray(vector)
#     np.save(label + '.npy', vector)


# In[4]:


# Train dataset loading and Stacking both label

X_train = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))


# In[5]:


X_train.shape


# In[6]:


# path = "C:/Users/NBH/Desktop/raw/dataset/test/10_FPS/30x30/"
# for label in labels:
#     vector=[]

#     images = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
#     for image in images:
#         img=cv2.imread(image,0)
# #         print(img)
#         vector.append(img)
#     vector = np.asarray(vector)
#     np.save(label + '.npy', vector)


# In[7]:


# Test dataset loading and  Stacking both label

X_test = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))


# In[8]:


#Normalizing

x_train = X_train.astype('float32')/255
x_test = X_test.astype('float32')/255

#Reshaping Data

x_train = x_train.reshape(x_train.shape[0],30,30,1)
x_test = x_test.reshape(x_test.shape[0],30,30,1)


# In[9]:


# Model Building

inp = Input((30,30,1))

layer = Conv2D(3, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(inp)
layer = MaxPooling2D(2,2) (layer)
layer = BatchNormalization()(layer)

layer = Dropout(0.7)(layer)

layer = Conv2D(3, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(layer)
layer = MaxPooling2D(2,2) (layer)
layer = BatchNormalization()(layer)

layer = Dropout(0.7)(layer)

layer = Conv2D(3, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(layer)
layer = MaxPooling2D(2,2) (layer)
layer = BatchNormalization()(layer)

layer = Dropout(0.7)(layer)

# layer = LSTM(200, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(layer)
# layer = BatchNormalization()(layer)

layer = Flatten() (layer)

layer = Dense(64, activation='relu')(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.8)(layer)


out = Dense(1, activation='sigmoid')(layer)

model = Model(input = inp, output = out)
model.summary()


# In[40]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train, epochs=500, verbose=1, validation_data=(x_test,y_test), batch_size=500, callbacks=[TensorBoard(log_dir='/tmp/cnn')])

