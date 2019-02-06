import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
# import tensorflow as tf
#from sklearn import SVM
import tensorboard
from time import time
from sklearn.metrics import classification_report


#min_max_scaler = preprocessing.normalize()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

path="/home/shakil/Desktop/violent_flow_detection/Data/"

labels=['violent','non_violent']
'''
for label in labels:
    vector=[]

    images = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
    for image in images:
        img=cv2.imread(image,0)
        vector.append(img)
    np.save(label + '.npy', vector)

'''

X_train = np.load('C:/Users/NBH/Desktop/raw/dataset/train/10 FPS/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/raw/dataset/train/10 FPS/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))

assert X_train.shape[0] == len(y_train)


X_test = np.load('C:/Users/NBH/Desktop/raw/dataset/test/10 FPS/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/raw/dataset/test/10 FPS/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))

assert X_test.shape[0] == len(y_test)

#x_train, x_test,y_train,y_test= train_test_split(X, y, test_size= .2, random_state=42, shuffle=True)

y_train_hot=to_categorical(y_train)
y_test_hot=to_categorical(y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


model=Sequential()

model.add(Conv1D(1,3, activation='relu', input_shape=(28,28)))
model.add(Conv1D(1,3, activation='relu'))

#model.add(CuDNNLSTM(10,return_sequences=True, kernel_regularizer=regularizers.l2(0.01), input_shape=(28,28)))
#model.add(Dropout(0.50))

model.add(CuDNNLSTM(50,return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.50))
model.add(BatchNormalization())
#model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#model.add(BatchNormalization())

#model.add(Flatten())
#model.add(Dense(10,activation='relu'))
model.add(Dense(100,activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.50))
model.add(Dense(10,activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.50))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train_hot, batch_size=100, epochs=50, verbose=1,validation_data=(X_test,y_test_hot), callbacks=[keras.callbacks.TensorBoard(log_dir="logs/lstm1/500_epocs{}".format(time()), histogram_freq=1, write_graph=False, write_images=True)]
)

#model.save_weights('movie_data.h5')

score,acc=model.evaluate(X_test, y_test_hot, batch_size=100)


    



