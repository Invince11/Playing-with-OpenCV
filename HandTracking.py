
# coding: utf-8

# In[1]:

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)




while(True):
    
    ret, frame = cam.read()
    
    if(not ret):
        print('Image not captured!')
        continue
        
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    
    
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    frame = cv2.flip(frame, 1)
    
    cv2.imshow('original', frame)
    
    
    img = frame
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)
    cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, 60, 255, 0)
    cv2.imshow('ori', thresh)
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur1 = cv2.GaussianBlur(hsv, (5,5), 0)
    
    lower_color = np.array([2,50,50])
    upper_color = np.array([15,225,225])
    skin_mask = cv2.inRange(blur1, lower_color, upper_color)
    
    skin_mask_copy = skin_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   
    opening = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
#     skin_mask = cv2.erode(skin_mask, kernel)
#     skin_mask = cv2.dilate(skin_mask, kernel)
#     skin_mask = cv2.dilate(skin_mask, kernel)
#     skin_mask = cv2.erode(skin_mask, kernel)
    
    blur2 = cv2.medianBlur(closing,5)
    
    ret,thresh = cv2.threshold(blur2,127,255,0)
    
    skin_mask_copy = cv2.bitwise_and(skin_mask,thresh)
  
    _,contours,_ = cv2.findContours(skin_mask_copy, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    
#     largest contour
    max_area = -999999
    c_index = 0
    
    for i in range(len(contours)):
        c = contours[i]
        area = cv2.contourArea(c)
        if(area > max_area):
            max_area = area
            c_index = i
    
    largest_contour = contours[c_index]
    hull = cv2.convexHull(largest_contour)
    
# #       display contour
#     x,y,w,h = cv2.boundingRect(largest_contour)
#     cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
#     cnt = thresh[y:y+h,x:x+w]

# draw on skin mask
    x,y,w,h = cv2.boundingRect(largest_contour)
    img = cv2.rectangle(skin_mask_copy,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.drawContours(skin_mask_copy,[hull],-1,(255,255,255),2)
    
# draw on frame
    x,y,w,h = cv2.boundingRect(largest_contour)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
  
    
    #show hull contour
    mask = np.zeros((img.shape[0], img.shape[1]))
    hull_contour_image = cv2.fillConvexPoly(mask, hull, 1) 
    
    
    
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    cv2.imshow('Threshold',skin_mask_copy)                                        

    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()   

