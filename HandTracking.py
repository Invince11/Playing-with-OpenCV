
# coding: utf-8

# In[1]:

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

def nothing(x):
    pass


HSV_TrackBar = np.zeros([100,700], np.uint8)

cv2.namedWindow("HSV_TrackBar")
cv2.createTrackbar('L_h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('L_s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('L_v', 'HSV_TrackBar',0,255,nothing)

cv2.createTrackbar('H_h', 'HSV_TrackBar',179,179,nothing)
cv2.createTrackbar('H_s', 'HSV_TrackBar',255,255,nothing)
cv2.createTrackbar('H_v', 'HSV_TrackBar',255,255,nothing)



while True:
    ret, frame = cam.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    
    L_h = cv2.getTrackbarPos('L_h', 'HSV_TrackBar')
    L_s = cv2.getTrackbarPos('L_s', 'HSV_TrackBar')
    L_v = cv2.getTrackbarPos('L_v', 'HSV_TrackBar')
    
    H_h = cv2.getTrackbarPos('H_h', 'HSV_TrackBar')
    H_s = cv2.getTrackbarPos('H_s', 'HSV_TrackBar')
    H_v = cv2.getTrackbarPos('H_v', 'HSV_TrackBar')
    
    
#     lower_color = np.array([L_h, L_s, L_v])
    lower_color = np.array([108, 23, 82])
    upper_color = np.array([H_h, H_s, H_v])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)
    
    
    _,contours,_ = cv2.findContours(hsv_d, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)   
    
    max_area = -10000
    c_index = 0
    
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area > max_area):
            max_area = area
            c_index = i
    
    largest_contour = contours[c_index]
    hull = cv2.convexHull(largest_contour)
    
    cv2.drawContours(frame,contours,c_index, (0,0,255),2)
    cv2.drawContours(frame,[hull],-1,(0,255,255),2)
  
    # defects (gap between fingers)
    hullIndices = cv2.convexHull(largest_contour, returnPoints = False)
    
    # defect[i] = [start_point, ,end_point, farthest_point, distance_From_farthest_point]
    defects = cv2.convexityDefects(largest_contour, hullIndices)
    
#     print(defects.shape)
    # detecting fingers (from documentation)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])
#         cv2.line(frame,start,end,[255,0,0],2)
        cv2.line(frame,start,far,[255,0,0],2)
        cv2.line(frame,far,end,[255,0,0],2)
        cv2.circle(frame,far,5,[0,255,0],-1)
     
    
    
    
    cv2. imshow('p', hsv_d)
    cv2.imshow('o', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



##alternate method below

# cam = cv2.VideoCapture(0)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)




# while(True):
    
#     ret, frame = cam.read()
    
#     if(not ret):
#         print('Image not captured!')
#         continue
        
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    
    
#     frame = cv2.bilateralFilter(frame, 5, 50, 100) 
#     frame = cv2.flip(frame, 1)
    
#     cv2.imshow('original', frame)
    
    
#     img = frame
    
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (41, 41), 0)
#     cv2.imshow('blur', blur)
#     ret, thresh = cv2.threshold(blur, 60, 255, 0)
#     cv2.imshow('ori', thresh)
    

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     blur1 = cv2.GaussianBlur(hsv, (5,5), 0)
    
#     lower_color = np.array([2,50,50])
#     upper_color = np.array([15,225,225])
#     skin_mask = cv2.inRange(blur1, lower_color, upper_color)
    
#     skin_mask_copy = skin_mask
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   
#     opening = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
# #     skin_mask = cv2.erode(skin_mask, kernel)
# #     skin_mask = cv2.dilate(skin_mask, kernel)
# #     skin_mask = cv2.dilate(skin_mask, kernel)
# #     skin_mask = cv2.erode(skin_mask, kernel)
    
#     blur2 = cv2.medianBlur(closing,5)
    
#     ret,thresh = cv2.threshold(blur2,127,255,0)
    
#     skin_mask_copy = cv2.bitwise_and(skin_mask,thresh)
  
#     _,contours,_ = cv2.findContours(skin_mask_copy, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    
# #     largest contour
#     max_area = -999999
#     c_index = 0
    
#     for i in range(len(contours)):
#         c = contours[i]
#         area = cv2.contourArea(c)
#         if(area > max_area):
#             max_area = area
#             c_index = i
    
#     largest_contour = contours[c_index]
#     hull = cv2.convexHull(largest_contour)
    
# # #       display contour
# #     x,y,w,h = cv2.boundingRect(largest_contour)
# #     cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
# #     cnt = thresh[y:y+h,x:x+w]

# # draw on skin mask
#     x,y,w,h = cv2.boundingRect(largest_contour)
#     img = cv2.rectangle(skin_mask_copy,(x,y),(x+w,y+h),(255,255,255),2)
#     cv2.drawContours(skin_mask_copy,[hull],-1,(255,255,255),2)
    
# # draw on frame
#     x,y,w,h = cv2.boundingRect(largest_contour)
#     img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.drawContours(frame,[hull],-1,(255,255,255),2)
  
    
#     #show hull contour
#     mask = np.zeros((img.shape[0], img.shape[1]))
#     hull_contour_image = cv2.fillConvexPoly(mask, hull, 1) 
    
    
    
#     cv2.drawContours(frame,[hull],-1,(255,255,255),2)
#     cv2.imshow('Threshold',skin_mask_copy)                                        

#     cv2.imshow('Frame',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindows()   

