from vtf import *
import os

path="C:/Users/NBH/Desktop/raw/raw/train/violent/"

output_location="C:/Users/NBH/Desktop/raw/dataset/train/violent/"

videos = [vfile for vfile in os.listdir(path)]

for video in videos:
    video_path=os.path.join(path,video)
    print("Current Videofile Name: ",video)
    video_to_frames(video_path,output_location,video)





