import cv2 as cv
import os
import time
import HandTrackingModule as htm

video = cv.VideoCapture(0)
video.set(3,1920)
video.set(4,1080)

detector = htm.HandDetector(detectionCon=0.75)
fingerids = [4, 8, 12, 16, 20]

while True:
    success,frame = video.read()
    img = cv.flip(frame,1)
    img = detector.FindHands(img)
    lmlist = detector.FindPosition(img, draw=False)
    print(lmlist)
    if(len(lmlist)!=0):
        fingers = []
        #Thumb (checking if tip of thumb is more left side then its below finger)
        if lmlist[fingerids[0]][1] > lmlist[fingerids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 fingers (checking if the fingers are bleow there -2 points)
        for i in range(1,5):
            if lmlist[fingerids[i]][2] < lmlist[fingerids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = fingers.count(1)
        print(total_fingers)
        print(fingers)
        cv.putText(img, str(total_fingers), (45,375), cv.FONT_HERSHEY_COMPLEX, 5, (0,0,255), 25)
            
    cv.imshow("Video", img)

    if(cv.waitKey(1)==ord('q')):
        break
