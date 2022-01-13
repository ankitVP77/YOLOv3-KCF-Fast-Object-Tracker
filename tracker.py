import cv2
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-v','--video',required=True,help='Path to the input video file.')
ap.add_argument('-c', '--cam', help='Flag to determine whether webcam should be used. Specify "Y" or "N"')
ap.add_argument('-o','--object',default="sports ball",help='The name of the object class to detect')
arg=vars(ap.parse_args())

objth=0.5
confth=0.5
nmsth=0.5
inwidth=320
inheight=320

classes=None
with open('coco.names','rt') as f:
    classes=f.read().rstrip('\n').split('\n')

model=cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
tracker=cv2.TrackerKCF_create()

def getoutputlayers(model):
    layernames=model.getLayerNames()
    return [layernames[i[0]-1] for i in model.getUnconnectedOutLayers()]


def processndraw(frame, out, timer):
    imghei = frame.shape[0]
    imgwid = frame.shape[1]
    classids = []
    confs = []
    boxes = []

    for o in out:
        for detected in o:
            if detected[4] > objth:
                scores = detected[5:]
                classid = np.argmax(scores)
                conf = scores[classid]
                if conf > confth and classes[classid] == arg['object']:
                    center_x = int(detected[0] * imgwid)
                    center_y = int(detected[1] * imghei)
                    w = int(detected[2] * imgwid)
                    h = int(detected[3] * imghei)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    classids.append(classid)
                    confs.append(float(conf))
                    boxes.append([x, y, w, h])
    ind = cv2.dnn.NMSBoxes(boxes, confs, confth, nmsth)
    for i in ind:
        i = i[0]
        box = boxes[i]
        cframe = frame.copy()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 0), 3)
        cv2.putText(frame, "FPS : " + str(int(fps)), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2);
        cv2.putText(frame, arg['object'], (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, "YOLO Detect", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.imshow('F', frame)
        outv.write(frame)
    if len(boxes) == 0:
        box = []
        cframe = frame.copy()
    return box, cframe

def ytrack(frame,model,outv,timer):
    global bflag
    blob=cv2.dnn.blobFromImage(frame,1/255,(inwidth,inheight),[0,0,0],1,crop=False)
    model.setInput(blob)
    out=model.forward(getoutputlayers(model))
    tbox,cframe=processndraw(frame,out,timer)
    return tbox,cframe

if arg['cam']== "Y":
    cap = cv2.VideoCapture(0)
else:
    cap=cv2.VideoCapture(arg['video'])
outv=cv2.VideoWriter("Tracked-"+str(arg['video']),cv2.VideoWriter_fourcc(*'MP4V'),int(cap.get(5)),(int(cap.get(3)),int(cap.get(4))))

count=0
tflag=0
fdflag=0
while(cap.isOpened()):
    ret,frame=cap.read()
    timer=cv2.getTickCount()
    if ret==True:
        if (count%30==0 or tflag==0) and fdflag!=1:
            tbox,cframe=ytrack(frame,model,outv,timer)
            if len(tbox)==0:
                fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
                cv2.putText(frame, "FPS : " + str(int(fps)), (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                cv2.putText(frame, "Tracking failure detected", (20,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                cv2.putText(frame,"YOLO Detect", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,0),2)
                cv2.imshow('F',frame)
                outv.write(frame)
            else:
                tracker=cv2.TrackerKCF_create()
                tracker.init(cframe,(tbox[0],tbox[1],tbox[2],tbox[3]))
                tflag=1
        elif count%30!=0 and tflag==1:
            rett,bbox=tracker.update(frame)
            fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
            if rett==True:
                tflag=1
                pt1=(int(bbox[0]),int(bbox[1]))
                pt2=(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
                cv2.rectangle(frame,pt1,pt2,(0,255,0),3)
                cv2.putText(frame, "FPS : " + str(int(fps)), (20,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
                cv2.putText(frame,arg['object'],(pt1[0]-5,pt1[1]-5),cv2.FONT_HERSHEY_COMPLEX,0.65,(255,255,0),2)
                cv2.putText(frame,"KCF Track", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
                cv2.imshow('F',frame)
                outv.write(frame)
            else:
                tflag=0
                tbox,cframe=ytrack(frame,model,outv,timer)
                if len(tbox)==0:
                    fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
                    cv2.putText(frame, "FPS : " + str(int(fps)), (20,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2);
                    cv2.putText(frame, "Tracking failure detected", (20,80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    cv2.putText(frame,"KCF Detect", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
                    cv2.imshow('F',frame)
                    outv.write(frame)
                else:
                    tracker=cv2.TrackerKCF_create()
                    tracker.init(cframe,(tbox[0],tbox[1],tbox[2],tbox[3]))
                    tflag=1
        count+=1
        if count==1:
            fdflag=1
        else:
            fdflag=0
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
    else:
        break
cap.release()
outv.release()
cv2.destroyAllWindows()

