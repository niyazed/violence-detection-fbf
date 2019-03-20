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


model = load_model("hockey_pretrain.h5")


#Freeze the first 6 layers
n_model = Sequential()

for layer in model.layers:
	n_model.add(layer) 

for layer in n_model.layers[:6]:
	layer.trainable = True

n_model.summary()

adam = optimizers.Adam(lr=0.001)
n_model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

n_model.fit(X_train,y_train_cat, epochs=100, verbose=1, validation_data=(X_test,y_test_cat), batch_size=300, callbacks=[TensorBoard(log_dir='/tmp/cnn')])