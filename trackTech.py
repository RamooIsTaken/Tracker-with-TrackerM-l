import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

tracker = cv2.TrackerMIL_create()

gt = pd.read_csv("gt_new.txt")
videoPAth = "MOT17-13-SDP.mp4"


cap = cv2.VideoCapture(videoPAth)

#Public parameters 
initBB = None # kutucuk için
fps = 25
frameNum =[]
f = 0
successTrackFrameSucces = 0
trackList = []
startTime = time.time() #süreyi başlatmak için

while True :
    time.sleep(0.01)

    ret ,frame = cap.read()

    if ret :
        frame = cv2.resize(frame,(960,540))
        (H,W) = frame.shape[:2]

        #Gt al
        carGt = gt[gt.frameNo == f]
        if len(carGt) != 0 :
            x = carGt.x.values[0]
            y = carGt.y.values[0]
            w = carGt.w.values[0]
            h = carGt.h.values[0]

            centerX = carGt.centerX.values[0]
            centerY = carGt.centerY.values[0]

            #çizim 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame,(centerX,centerY),2,(0,0,255),cv2.FILLED)

        # box 
        if initBB is not None : 
            (success, box) = tracker.update(frame)
            
            if f<= np.max(gt.frameNo) : # araç açıdan çıktığı zaman takibi kesmek için
                (x,y,w,h) = [int(i) for i in box ]
                cenX  = int(x+(w/2))
                cenY = int(y+(h/2))
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.circle(frame,(cenX,cenY),2,(0,255,0),4,-1)
                successTrackFrameSucces = successTrackFrameSucces + 1 
                trackCenterX = int(x+y/2)
                trackCenterY = int(y+h/2)
                trackList.append([f,trackCenterX,trackCenterY]) # gt ile takip algoritmasının centerları karşılaştırmak için kullanacağız
                

            info = [("Tracker","TrackerMIL"),
                    ("Success","Yes" if success else "No")]
            
            for (i,(o,p)) in enumerate(info) :
                text =  "{} : {}".format(o,p)
                cv2.putText(frame,text,(10,H - (i*20) - 10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            
        cv2.putText(frame,"Frame Num : "+str(f),(10,30),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("t") :
            initBB = cv2.selectROI("ROI SEC",frame,fromCenter=False,)
            tracker.init(frame,initBB)

        elif key == ord("q") : break

        frameNum.append(f)
        f = f +1
    
    else : break

cap.release()
cv2.destroyAllWindows()




# değerlendirme
stop = time.time()
timeDiff = stop - startTime

trackDf = pd.DataFrame(trackList,columns=["frameNo","centerX","centerY"])

if len(trackDf) != 0 :
    print("Time : ", timeDiff)
    print("Number Of Frame to Track (gt) :" ,len(gt))
    print("Number Of Frame to Track (track Succ) :" ,successTrackFrameSucces)

    trackDfFrame = trackDf.frameNo 
    gt_centerX = gt.centerX[trackDfFrame].values
    gt_centerY = gt.centerY[trackDfFrame].values

    trackDfCenterX = trackDf.centerX.values
    trackDfCenterY = trackDf.centerY.values

    error = (np.sqrt((gt_centerX-trackDfCenterX)**2 +(gt_centerY - trackCenterY)**2))
    print("Toplam hata : ",error)
    plt.figure()
    plt.plot(np.sqrt((gt_centerX-trackDfCenterX)**2 +(gt_centerY - trackCenterY)**2))
    plt.xlabel("Frame")
    plt.ylabel("Oklid mesafesi gt-track")
    plt.show()