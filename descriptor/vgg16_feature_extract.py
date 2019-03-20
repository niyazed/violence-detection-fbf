# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# import numpy as np
# import matplotlib.pyplot as plt

# img_path = 'violentframe0 (1).jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# model = VGG16(weights='imagenet', include_top=True)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# features = model.predict(x)
# model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
# fc2_features = model_extractfeatures.predict(x)
# fc2_features = fc2_features.reshape((4096,1))
# np.savetxt('fc2.txt',fc2_features)
# plt.plot(fc2_features)
# plt.show()


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers import *
import pylab
from keras.models import Model	
import matplotlib.pyplot as plt


frames = np.load('violent.npy')

print('Frame SHAPE: ',frames.shape)

model = VGG16(input_shape=(224,224,3),weights='imagenet', include_top=False)

new_model = model.output

# new_model = GlobalAveragePooling2D() (new_model)

new_model = Model(input=model.input,output=new_model)

# new_model.summary()

# img_path = 'violentframe0 (1).jpg'
# img = image.load_img(img_path)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
desc = preprocess_input(frames)

print(' Desc SHAPE: ',desc.shape)



features = new_model.predict(desc,batch_size=10,verbose=1)
print (features.shape)

pic=features[0,:,:,1]
pylab.imshow(pic)
pylab.gray()
pylab.show()