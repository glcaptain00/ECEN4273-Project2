import sys          #For reading Command Line arguments
import cv2 as cv    #For opening and processing images and videos
import traceback    #For printing excpetion trace


# Configurables
# No configurables yet

############################
# Video Processing Section #
############################
def openImage(src):
    return cv.imread(src)

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

if (len(sys.argv) < 3):
    print("Usage: python main.py <video source type> <file name>\nWhere type can be:\n\t-live: file name will be an integer representing the camera to stream video from\n\t-img: Use an image file\n\t-video: Use a video file")
    sys.exit()

sourceType = sys.argv[1]
videoSource = sys.argv[2]

if (sourceType == "-live"):
    print("Live not yet implemented")
    sys.exit()
elif (sourceType == "-img" or videoSource == "-image"):
    im = cv.imread(videoSource)

    newIm = cv.pyrDown(im)
    ret, newIm = cv.threshold(cv.cvtColor(newIm, cv.COLOR_BGR2GRAY), 64, 255, cv.THRESH_BINARY)
    cv.imshow("Frame", newIm)
    cv.waitKey(0)

elif (sourceType == "-video"):

    myVid = openVideo(videoSource)
    myFrames = splitFrames(myVid)

    #cv.rectangle(myFrames[0], (10, 10), (30,30), (255,0,0), 2)

    newImg = cv.pyrDown(myFrames[0])
    ret, newImg = cv.threshold(cv.cvtColor(newImg, cv.COLOR_BGR2GRAY), 64, 255, cv.THRESH_BINARY)
    cv.imshow("Frame", newImg)
    cv.waitKey(0)

