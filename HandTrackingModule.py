import mediapipe as mp
import cv2 as cv
import time

class HandDetector():  #constructor
    def __init__(self, mod = False, maxhands = 2, modelCom = 1,detectionCon = 0.5, trackCon = 0.5):

        self.mod = mod
        self.maxhands = maxhands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.cTime = self.pTime = self.fps = 0

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mod, self.maxhands, self.modelCom, self.detectionCon, self.trackCon) #calling statically
        self.mpdraw = mp.solutions.drawing_utils

        self.fingerids = [4, 8, 12, 16, 20]

    def FindHands(self, img, draw = True): #method for finding hands

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if (self.results.multi_hand_landmarks):
            for handLndMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handLndMs,self.mphands.HAND_CONNECTIONS)
        return img

    def FindPosition(self, img, handNo=0,draw=True):  #method for locating points of tips of hands

        self.lmlist = []

        if (self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ld in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(ld.x*w), int(ld.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy),5,(0,255,0), cv.FILLED)
        return self.lmlist

    def Calcfps(self, img, draw=True):  #method for calculating FPS
        self.cTime = time.time()
        self.fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime
        if draw:
            cv.putText(img, str(int(self.fps)), (10,40), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return self.fps

    def fingersup(self):   #counting Fingers
        fingers = []
        if self.lmlist[self.fingerids[0]][1] > self.lmlist[self.fingerids[0]-1][1]:      
            fingers.append(1)
        else:
            fingers.append(0)
        
        for i in range(1,5):
            if self.lmlist[self.fingerids[i]][2] < self.lmlist[self.fingerids[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = fingers.count(1)

        return fingers

def main():
    vid = cv.VideoCapture(0)
    detector = HandDetector()  #object of the class
    while True:
        yes,frame = vid.read()
        video_flip = cv.flip(frame, 1)
        img = detector.FindHands(video_flip)  #calling FindHands methof from the class
        pos = detector.FindPosition(img)
        fps = detector.Calcfps(img)
        print()
        if (len(pos) != 0):
            fingers = detector.fingersup()
            print(fingers)
        cv.imshow("video",img)
        if(cv.waitKey(1)==ord('q')):
            break

if __name__ == "__main__":
    main()