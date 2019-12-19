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
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import VGG16


labels=['violent','non_violent']


# Train dataset loading and Stacking both label

X_train = np.load('E:/Niloy/RGB224x224/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('E:/Niloy/RGB224x224/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))


# Test dataset loading and  Stacking both label

X_test = np.load('E:/Niloy/RGB224x224/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('E:/Niloy/RGB224x224/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))
    

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# y_test = to_categorical(y_test,2)
# y_train = to_categorical(y_train,2)
#Reshaping Data

X_train = X_train.reshape(X_train.shape[0],224,224,3)
X_test = X_test.reshape(X_test.shape[0],224,224,3)


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in vgg_conv.layers:
    layer.trainable = False
 

for layer in vgg_conv.layers:
    print(layer, layer.trainable)


model = Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(1, activation='sigmoid'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,y_train, epochs=3, verbose=1, validation_data=(X_test,y_test), batch_size=30)

model.save('vgg16_model.h5')
