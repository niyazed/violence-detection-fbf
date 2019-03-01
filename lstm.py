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
from keras import layers
from keras import Model
from keras.callbacks import TensorBoard
from keras import optimizers


labels=['violent','non_violent']


# Train dataset loading and Stacking both label

X_train = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))



# Test dataset loading and  Stacking both label

X_test = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))



#Normalizing

#x_train = X_train.astype('float32')/255
#x_test = X_test.astype('float32')/255





#MODEL
model=Sequential()

model.add(Conv1D(1,3, activation='relu', input_shape=(30,30)))
model.add(MaxPooling1D(2,2))

model.add(Conv1D(1,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(2,2))
# model.add(BatchNormalization())

model.add(Conv1D(1,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(2,2))
# model.add(BatchNormalization())

# model.add(Conv1D(1,3))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(CuDNNLSTM(200,return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))

# model.add(BatchNormalization())

model.add(Dense(100,activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.80))
model.add(Dense(10,activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.80))
model.add(Dense(1, activation='sigmoid'))
model.summary()



rmsprop = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.001)

model.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(X_train,y_train, epochs=20000, verbose=1, validation_data=(X_test,y_test), batch_size=500, callbacks=[TensorBoard(log_dir='/tmp/cnn')])