import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns



colLıst = ["frame_number","identity_number","left","top","width","height","score","class","visibility"]

data = pd.read_csv("gt.txt",names=colLıst)

car = data[data["class"] == 3] # araçların class ıd ler 3 makalede yazıyor farklı cisimler için farklı id ler mevcut

videoPath = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(videoPath)
id1 = 29 
numberOfImage = np.max(data["frame_number"])
fps = 25
boundBoxList = []

for i in range(numberOfImage-1) :
    ret ,frame = cap.read()

    if ret :
        frame = cv2.resize(frame,dsize=(960,540))

        filterId1 = np.logical_and(car["frame_number"] == i+1,car["identity_number"] == id1)
        
        if len(car[filterId1]) != 0:
            x = int(car[filterId1].left.values[0]/2)
            y = int(car[filterId1].top.values[0]/2)
            w = int(car[filterId1].width.values[0]/2)
            h = int(car[filterId1].height.values[0]/2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame,(int(x+w/2),int(y+h/2)),2,(0,0,255),1)
            #frame,x,y,genişlik,yüksek,centerX,centerY
            boundBoxList.append([i,x,y,w,h,int(x+w/2),int(y+h/2)])
        
        cv2.putText(frame,"Frame Num : "+str(i+1),(10,30),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
        cv2.imshow("Frame",frame)

        if cv2.waitKey(1) & 0xFF ==ord("q") : break
    else :break
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(boundBoxList,columns=["frameNo","x","y","w","h","centerX","centerY"])
df.to_csv("gt_new.txt",index=False)