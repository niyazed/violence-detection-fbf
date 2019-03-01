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


inp = Input((30,30))
layer = Conv1D(1,3, kernel_regularizer=regularizers.l2(0.01))(inp)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer) 

layer = Conv1D(1,3, kernel_regularizer=regularizers.l2(0.01))(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer)


layer = CuDNNLSTM(128,return_sequences=True, kernel_regularizer=regularizers.l2(0.01)) (layer)
layer = Dropout(0.8)(layer)

layer = CuDNNLSTM(128, kernel_regularizer=regularizers.l2(0.01)) (layer)
layer = Dropout(0.8)(layer)


layer = Dense(64)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer)
layer = Dropout(0.8)(layer)

layer = Dense(16)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer)
layer = Dropout(0.8)(layer)

out = Dense(1, activation='sigmoid')(layer)

model = Model(input = inp, output = out)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(X_train,y_train, epochs=10000, verbose=1, validation_data=(X_test,y_test), batch_size=500, callbacks=[TensorBoard(log_dir='/tmp/cnn')])

