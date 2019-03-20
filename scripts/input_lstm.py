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

X_train = np.load('C:/Users/NBH/Desktop/violence-detection/28x28/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/28x28/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))



# Test dataset loading and  Stacking both label

X_test = np.load('C:/Users/NBH/Desktop/violence-detection/28x28/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/28x28/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))

print(X_train.shape,X_test.shape)

#Normalize
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)
# print(y_test_cat)

inp = Input((28,28))


# layer = CuDNNLSTM(2048,return_sequences=False, kernel_regularizer=regularizers.l2(0.01)) (inp)
layer = Flatten() (inp)
layer = Dense(2048) (layer)
layer = Dropout(0.8)(layer)

layer = Dense(512)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer)
layer = Dropout(0.5)(layer)

out = Dense(2, activation='softmax')(layer)

model = Model(input = inp, output = out)
model.summary()

# adam = optimizers.Adam(lr=0.001,decay=0.1)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train,y_train_cat, epochs=200, verbose=1, validation_data=(X_test,y_test_cat), batch_size=70, callbacks=[TensorBoard(log_dir='/tmp/cnn')])
# model.save_weights("movies_weights.h5")