#ffmpeg -i input.mp4 -c copy -map 0 -segment_time 00:20:00 -f segment -reset_timestamps 1 output%03d.mp4

import os

def main():
    files = os.listdir("./")
    for f in files:
        if f.lower()[-3:] == "mp4":
            print "processing", f
            process(f)

def process(f):
    inFile = f
    outFile = f[:-4] 
    cmd = "ffmpeg -i {} -c copy -map 0 -segment_time 00:00:05 -f segment -reset_timestamps 1 {}%04d.mp4".format(inFile, outFile)
    os.popen(cmd)

main()
