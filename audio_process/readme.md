This folder is used to store audio processing codes

All the python scripts here requires python2.7+  // won't work with python3

#this requires ffmpeg
* audioex.py  = Extracts audio from video file
* videosplitter.py = Splits videos in segments mentioned in segments paramater

#this requires yaafe 0.70
Yaafe is a audio feature extraction tool. Learn more here: http://yaafe.github.io/Yaafe/
Installation instruction provided here: yaafe.github.io/Yaafe/manual/install.html

Note:We had to modify a few things from installation instruction which we will add later after more research

*directory : feature_extractor
	*featureextractor.py = extracts features defined in "featureplan" file from all files listed in "input" file
	*featureplan = Used to define features to be extracted. We have used this to configure features and configuration that we have used.
