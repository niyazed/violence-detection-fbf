#yaafe -c featureplan -r 44100 -i input

import os

def main():
	process()	

def process():
	cmd = "yaafe -c featureplan -r 44100 -i input"
	os.popen(cmd)

main()
