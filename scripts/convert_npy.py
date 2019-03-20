import numpy as np
import os
import cv2


labels=['violent','non_violent']
path = "C:/Users/NBH/Desktop/raw/dataset/test/10_FPS/32x32/"
for label in labels:
    vector=[]

    images = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
    for image in images:
        img=cv2.imread(image,0)
        #print(img)
        vector.append(img)
    vector = np.asarray(vector)
    np.save(label + '.npy', vector)
