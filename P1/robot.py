import cv2
import numpy as np
from HAL import HAL
from GUI import GUI

# Enter sequential code!
K_P = 0
K_I = 0
K_D = 0

FILTRO1 = np.array([[0, 100, 20],[10, 255, 255]])
FILTRO2 = np.array([[160, 50, 50],[180, 255, 255]])

while True:
    # Enter iterative code!
    im = HAL.getImage()

    # Convertir a HSV
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Aplicar rango
    mask1 = cv2.inRange(hsv_im, FILTRO1[0], FILTRO1[1])
    mask2 = cv2.inRange(hsv_im, FILTRO2[0], FILTRO2[1])
    mask = mask1 + mask2
    
    # Erode / dilate
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Escoger mayor blob
    n_labels, label_ids, values, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    index = np.argsort(values[:,cv2.CC_STAT_AREA])[-2]
    mask = (label_ids == index).astype(np.dtype("uint8")) * 255
    
    moment = cv2.moments(mask)
    if moment["m00"] != 0:
        center_coordinates = (int(moment["m10"]/moment["m00"]),int(moment["m01"]/moment["m00"]))
        cv2.circle(im, center_coordinates, 1, [0,255,0], 5)
        e = 329 - center_coordinates[0]
        u = e*Kp
        HAL.setV(2)
        HAL.setW(u)
    else:
        center_coordinates = (0,0)
        HAL.setV(0)
        HAL.setW(0.5)
    cv2.line(im, (int(im.shape[1]/2),0), (int(im.shape[1]/2),im.shape[0]-1), [255,0,0], 2)
    GUI.showImage(im)
