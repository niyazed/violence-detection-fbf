import os

def main():
	process()	

def process():
	cmd = "python2 ./videosplitter.py"
	os.popen(cmd)
	cmd = "audioex.py"
	os.popen(cmd)

main()
