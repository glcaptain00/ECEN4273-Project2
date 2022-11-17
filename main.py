import sys          #For reading Command Line arguments
import cv2 as cv    #For opening and processing images and videos
import traceback    #For printing excpetion trace


# Configurables
# No configurables yet

############################
# Video Processing Section #
############################
def openVideo(video):
    return cv.VideoCapture(video)

def splitFrames(video):
    frames = []
    success, fr = video.read()
    if (success):
        frames.append(fr)
    else:
        traceback.print_exc()
        #raise Exception("Failed to read a frame!")
    
    return frames







########
# Main #
########

if (len(sys.argv) < 2):
    print("Usage: python main.py <video source>")
    sys.exit()

videoSource = sys.argv[1]

myVid = openVideo(videoSource)
myFrames = splitFrames(myVid)

cv.imshow("Frame", myFrames[0])
cv.waitKey(0)

