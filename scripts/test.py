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

X_train = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))


X_test = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/30x30_10FPS/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))


n_frames = 10

X_train = X_train.reshape(X_train.shape[0],30,30,1)
X_test = X_test.reshape(X_test.shape[0],30,30,1)

# X_train = np.expand_dims(X_train, axis=0)

print(X_train.shape)

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(n_frames, 30, 30, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

# seq.add(Flatten())

seq.add(Dense(1, activation='sigmoid'))

seq.summary()

seq.compile(loss='binary_crossentropy', optimizer='adam')




seq.fit(X_train,y_train, epochs=500, verbose=1, validation_data=(X_test,y_test), batch_size=50, callbacks=[TensorBoard(log_dir='/tmp/cnn')])