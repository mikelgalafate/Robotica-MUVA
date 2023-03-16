from GUI import GUI
from HAL import HAL
import cv2
import numpy as np
# Enter sequential code!
Kp = 2.2e-3
v = 4
filtro = np.array([[17,15,100],[50,56,255]])

while True:
    # Enter iterative code!
    im = HAL.getImage()
    
    mask = cv2.inRange(im, filtro[0], filtro[1])
    
    kernel = np.ones((5, 5), np.uint8)
    
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    
    # Quedarse con el más grande
    out = cv2.bitwise_and(im, im, mask = mask)
    
    moment = cv2.moments(mask)
    if moment["m00"] != 0:
        center_coordinates = (int(moment["m10"]/moment["m00"]),int(moment["m01"]/moment["m00"]))
        cv2.circle(im, center_coordinates, 1, [0,255,0], 5)
        e = im.shape[1]/2 - center_coordinates[0] + 5
        u = e*Kp
        #HAL.setV(1/(1+np.abs(u*1.5))*v)
        #HAL.setW(u)
        print(center_coordinates)
    else:
        center_coordinates = (0,0)
        HAL.setV(0)
        HAL.setW(0.5)
    cv2.line(im, (int(im.shape[1]/2),0), (int(im.shape[1]/2),im.shape[0]-1), [255,0,0], 2)
    GUI.showImage(im)
    