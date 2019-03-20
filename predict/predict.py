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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import math
import time
import numpy as np
import os
import cv2



video_input="video/"

frame_output="frames/"

numpy_output = "numpy/"



video = os.listdir(video_input)
video = ''.join(video)
video_path=os.path.join(video_input,video)
# print(video_path)

def video_to_frames(input_loc, output_loc,video_name):

    count = 0
   
    cap = cv2.VideoCapture(input_loc)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    # print (frameRate)

    while(cap.isOpened()):
        frameId = cap.get(10) #current frame number
        # print('FRAME:', frameId)
        ret, frame = cap.read()
        
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frame = cv2.resize(frame,(224,224)) #resize frames
            filename = video_name + "frame%d.jpg" % count;count+=1
            cv2.imwrite(output_loc + filename, frame)
            
    cap.release()
    print ("Frame Extraction Successful!")


video_to_frames(video_path,frame_output,video)


def convert_npy(input_loc,output_loc):
	path = input_loc
	vector=[]
	images = [path + '/' + wavfile for wavfile in os.listdir(path + '/')]
	for image in images:
		img=cv2.imread(image,1)
		vector.append(img)
	vector = np.asarray(vector)
	np.save(output_loc +'test' + '.npy', vector)
	print("Numpy Done!")


convert_npy(frame_output,numpy_output)

X_test = np.load('numpy/test.npy')

model = load_model('vgg16_28x28.h5')

result = model.predict(X_test,batch_size=20, verbose=1)

pred = []
for prediction in result:
	if prediction >.6:
		pred.append(1.0)
	else:
		pred.append(0.0)
# print(len(pred))
violent_count = 0
non_violent_count = 0
for number in pred:
	if(number == 0):
		violent_count += 1
	else:
		non_violent_count += 1

violence = (violent_count / len(pred))*100
non_violence = 100 - violence

print('Violence: %.2f%%' %violence)
print('Non violence: %.2f%%' %non_violence)
# result = 100 - np.round((result*100),2)
# print(np.mean(result))

# print(confusion_matrix(y_test,pred))
# print(pred[20])

def delete_files(video_loc, frames_loc, numpy_loc):
	videos = os.listdir(video_loc)
	# print(videos)
	frames = os.listdir(frames_loc)
	numpy = os.listdir(numpy_loc)
	# print(frames)
	for video in videos:
		video = os.path.join(video_loc, video)
		os.remove(video)
	print('Video deleted!')
	
	for frame in frames:
		frame = os.path.join(frames_loc,frame)
		os.remove(frame)
	print('Images deleted!')

	for numpy_file in numpy:
		numpy_file = os.path.join(numpy_loc,numpy_file)
		os.remove(numpy_file)
	print('numpy deleted!')

delete_files(video_input, frame_output, numpy_output)