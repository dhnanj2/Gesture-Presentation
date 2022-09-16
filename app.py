from flask import Flask
import cv2
import os 
import numpy as np
from HandTrackingModule import HandDetector

app = Flask(__name__)
@app.route('/home')

def index():
    # variables
    width, height = 1024, 576                        #dimension of main screen
    folderPath = "ppt"                               #path of the slides for presentation
    imgNum = 0                                       #counter/iterator for the current slide
    prev_rpinky_x = width                            #prvious x-coordinate of PINKY_PIP of right hand
    prev_lpinky_x = 0                                #prvious x-coordinate of PINKY_PIP of left hand
    annotations = [[]]                               #list of annotations   
    annotationNum = -1                               #index of current annotation   
    annotationStart = False                          #flag for annotation
    eraseCnt = 0                                     #counter for eraser

    # camera setup
    cap  = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    #check if the folder exists
    if not os.path.exists(folderPath):
        # if folder does not exist, than create one
        os.makedirs(folderPath)
    # get the list of presentation images
    pathImages = sorted(os.listdir(folderPath))

    #if pathImages[] is empty than exit
    if not pathImages: 
       print("No slide to present!!\nPlease add some slides in ~/GesturePresentation/ppt")
       quit() 

    # Hand Detector
    # detectionCon=0.8 means that consider it as hands if you are atleast 80% sure
    # maxHands=1 means max 1 hand will be detected
    detector = HandDetector(detectionCon=0.8, maxHands=1) 

    while True:
        # Import Images
        success, img = cap.read()

        # flip image (1 for horizontally and 0 for vetically)
        img = cv2.flip(img, 1)

        # genrate path for the current slide
        pathFullImage = os.path.join(folderPath, pathImages[imgNum])

        # read image for current slide
        imgCurrent = cv2.imread(pathFullImage)

        # detect hands 
        hands, img = detector.findHands(img)

        # resize the read image
        imgCurrent = cv2.resize(imgCurrent, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        # Decode hand Gesture if hands are detected
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)  # binary list of raised fingers
            lmList = hand['lmList']             # list of positions of 21 hand landmarks

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [width//4, width], [0, 2.5*width]))
            yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
            indexFinger = xVal, yVal

            # gesture 1 : Turn Left (move to next slide)
            # hand gesture = swipe right hand towards left <---
            if lmList and (fingers==[1,1,1,1,1] or fingers==[0,1,1,1,1]) and hand['type']=='Left': 
                cur_lpinky_x = lmList[19][0] # current x coordinate of PINKY_PIP 

                # if left swipe is greater than threshold than change slide number
                if cur_lpinky_x - prev_lpinky_x <= -125: 
                    imgNum = (imgNum+1)%len(pathImages)
                    # erase the previous annotations
                    annotations = [[]]                                  
                    annotationNum = -1                               
                    annotationStart = False

                prev_lpinky_x=cur_lpinky_x
                prev_rpinky_x = width

            # gesture 2 : Turn Right (move to previous slide)
            # hand gesture = swipe Left hand towards Right --->
            if lmList and (fingers==[1,1,1,1,1] or fingers==[0,1,1,1,1]) and hand['type']=='Right': 
                cur_rpinky_x = lmList[19][0] # current x coordinate of PINKY_PIP 

                # if left swipe is greater than threshold than change slide number
                if cur_rpinky_x - prev_rpinky_x >= 125: 
                    imgNum = imgNum-1 if imgNum>0 else len(pathImages)-1
                    # erase the previous annotations
                    annotations = [[]]                                  
                    annotationNum = -1                               
                    annotationStart = False
                prev_rpinky_x=cur_rpinky_x
                prev_lpinky_x = 0

            # gesture 3 : pointer
            # hand gesture = 2 fingers(index and middle) raised
            if fingers==[0,1,1,0,0] or fingers==[1,1,1,0,0]: 
                cv2.circle(imgCurrent, indexFinger, 5, (0,0,255), cv2.FILLED)

            # gesture 4 : draw/annotate
            # hand gesture = index finger raised
            if fingers==[0,1,0,0,0]: 
                if not annotationStart:
                    annotationStart = True
                    annotationNum += 1
                    annotations.append([])
                cv2.circle(imgCurrent, indexFinger, 5, (0,0,255), cv2.FILLED)
                annotations[annotationNum].append(indexFinger)

            else:
                annotationStart = False

            # gesture 5 : erase annotations
            # hand gesture = 3 finger (index, middle and ring) raised
            if fingers==[0,1,1,1,0]: 
                if annotations and eraseCnt>=5:
                    annotations.pop(-1)
                    annotationNum = annotationNum-1 if annotationNum>=0 else -1
                    eraseCnt=0
                else : eraseCnt+=1

        # executing stored annotations
        for annotation in range(len(annotations)):
            for i in range(len(annotations[annotation])):
                if i>0:
                    cv2.line(imgCurrent,annotations[annotation][i-1],annotations[annotation][i],(0,0,255),5)

        #adding webcam image on the slide
        h, w, _ = imgCurrent.shape
        ws,hs = w//6, h//5              #dimensions of subsceen(face cam)
        imgSmall = cv2.resize(img, (ws, hs))
        imgCurrent[0:hs, w-ws:w] = imgSmall

        # show slide/image
        cv2.imshow("Gesture Controlled Presentation (Press Esc to exit)", imgCurrent)
        key = cv2.waitKey(1)

        # if key pressed is 'ESC' then exit
        if key == 27: break

if __name__=="__main__":
    app.run()
