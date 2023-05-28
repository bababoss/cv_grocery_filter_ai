from __future__ import print_function
import cv2 
import argparse
import numpy as np

from classifier import inference
from common import pi_utils



def run_detector(model_net,video_path=None):
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video3.mp4')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args(args=[])

    kernel = np.ones((3,3),np.uint8)
    kernel_l = np.ones((7,7),np.uint8)
    fontScale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # Blue color in BGR
    color = (255, 0, 0)

    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (3840, 2160))

    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)
    i=0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fgMask = backSub.apply(frame)

        et,fgMask = cv2.threshold(fgMask,130,255,cv2.THRESH_BINARY)

        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_l)

        contours_list, hierarchy = cv2.findContours(image=fgMask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        bounding_rect = []
        thickness=1
        H,W,C=frame.shape
        for idx,cntr in enumerate(contours_list):
            #print(cntr.shape, cntr)
            x,y,w,h = cv2.boundingRect(cntr)
            if(w*h < 30):
                continue
            approx = cv2.approxPolyDP(cntr,0.01*cv2.arcLength(cntr,True),True)
            area = cv2.contourArea(cntr)
            if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
                if (x+h/2 > W/2-300) and (x+h/2 < W/2+300):
                    fg_count=fgMask[y:y+h,x:x+w].sum()/255
                    if fg_count >35000:
                      
                        bounding_rect.append([x,y,w,h])
                        crop_img = frame[y:y+h, x:x+w]
                        if i%10 ==0:
                            object_class=inference.predict(crop_img,model_net)
                            print(i,f"Predected class: {object_class}")
                            # Write here code to send to raspbery Pi
                            pi_utils.sent_signal_to_pi(object_class.lower())
                            cv2.imwrite(f"input/train/tomoto/c{i}_{idx}.jpg",crop_img)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness)
                        frame[y:y+h,x:x+w,1] = np.bitwise_or(frame[y:y+h,x:x+w,1], fgMask[y:y+h,x:x+w])

    #     cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #     cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        out.write(frame)
        
        if i%100 == 0:
            print("----",frame.shape)
            cv2.imwrite(f"data/output1/{i}.jpg",frame)
        if i==100000:
            break
        i+=1

        
        
        

    
    
