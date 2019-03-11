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

X_train = np.load('C:/Users/NBH/Desktop/violence-detection/32x32_10FPS/train/'+labels[0] + '.npy')
y_train = np.zeros(X_train.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/32x32_10FPS/train/'+label + '.npy')
    X_train = np.vstack((X_train, x))
    y_train = np.append(y_train, np.full(x.shape[0], fill_value= (i + 1)))



# Test dataset loading and  Stacking both label

X_test = np.load('C:/Users/NBH/Desktop/violence-detection/32x32_10FPS/test/'+labels[0] + '.npy')
y_test = np.zeros(X_test.shape[0])

for i, label in enumerate(labels[1:]):
    x = np.load('C:/Users/NBH/Desktop/violence-detection/32x32_10FPS/test/'+label + '.npy')
    X_test = np.vstack((X_test, x))
    y_test = np.append(y_test, np.full(x.shape[0], fill_value= (i + 1)))

print(X_train.shape,X_test.shape)

#Normalize
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)
# print(y_test_cat)

inp = Input((32,32))
layer = Conv1D(2,3)(inp)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer) 

# layer = Conv1D(2,3, kernel_regularizer=regularizers.l2(0.01))(layer)
# layer = BatchNormalization()(layer)
# layer = Activation('relu') (layer)

# # layer = Conv1D(1,3, kernel_regularizer=regularizers.l2(0.01))(layer)
# # layer = BatchNormalization()(layer)
# # layer = Activation('relu') (layer)

# layer = Dense(10)(layer)
# layer = BatchNormalization()(layer)
# layer = Activation('relu') (layer)
# layer = Dropout(0.5)(layer)


layer = CuDNNLSTM(50,return_sequences=False, kernel_regularizer=regularizers.l2(0.01)) (layer)
layer = Dropout(0.5)(layer)

# layer = CuDNNLSTM(50,return_sequences=False) (layer)
# layer = Dropout(0.5)(layer)
# layer = CuDNNLSTM(100,return_sequences=False, kernel_regularizer=regularizers.l2(0.01)) (layer)
# layer = Dropout(0.5)(layer)

# layer = CuDNNLSTM(200, kernel_regularizer=regularizers.l2(0.01)) (layer)
# layer = Dropout(0.5)(layer)


# layer = Activation('relu') (layer)
# layer = Dense(10, kernel_regularizer=regularizers.l2(0.01))(layer)
# layer = BatchNormalization()(layer)
# layer = Dropout(0.8)(layer)

layer = Dense(32)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu') (layer)
layer = Dropout(0.5)(layer)

out = Dense(2, activation='softmax')(layer)

model = Model(input = inp, output = out)
model.summary()

model.load_weights("movies_weights.h5")

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


model.fit(X_train,y_train_cat, epochs=100, verbose=1, validation_data=(X_test,y_test_cat), batch_size=32, callbacks=[TensorBoard(log_dir='/tmp/cnn')])
model.save_weights("movies_weights.h5")