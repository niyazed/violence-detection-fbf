import os

def main():
    files = os.listdir("./")
    for f in files:
        if f.lower()[-3:] == "mp4":
            print "processing", f
            process(f)

def process(f):
    inFile = f
    outFile = f[:-3] + "mp3"
    cmd = "ffmpeg -i {} -vn  -ac 2 -ar 44100 -ab 320k -f mp3 {}".format(inFile, outFile)
    os.popen(cmd)

main()
