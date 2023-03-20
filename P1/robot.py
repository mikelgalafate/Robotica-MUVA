import cv2
import numpy as np
from HAL import HAL
from GUI import GUI

# Enter sequential code!
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale =  0.5
color =  [0, 0, 255]
thickness =  1
(_, textSize), _ = cv2.getTextSize("Test", font, fontScale, thickness)

errors = np.zeros((1, 15))

K_P = 10.5e-3
K_I = 1.5e-4
K_D = 12.5e-3
V_MAX = 10


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
    if n_labels == 1:
        HAL.setV(0)
        HAL.setW(-0.5*np.sign(errors[0,1]))
    else:
        index = np.argsort(values[:,cv2.CC_STAT_AREA])[-2]
        
        mask = (label_ids == index).astype(np.dtype("uint8")) * 255
        
        offset = (values[index, cv2.CC_STAT_HEIGHT] - 1) % 5
        start = values[index, cv2.CC_STAT_TOP] + offset
        
        a = int(np.mean(np.where(mask[start,:] > 0)))
        e = np.sum((np.array(a) - im.shape[1] / 2))
        
        errors = np.roll(errors, 1)
        errors[0,0] = e
        up = - K_P * e
        ui = - K_I * np.sum(errors)
        ud = - K_D *(e - errors[0,1]) 
        u = up + ui + ud
        v = 1/(1+np.abs(u*1.5)) * V_MAX
        HAL.setV(v)
        HAL.setW(u)
        
        cv2.line(im, (int(im.shape[1]/2),values[index, cv2.CC_STAT_TOP]), (int(im.shape[1]/2), values[index, cv2.CC_STAT_TOP] + values[index, cv2.CC_STAT_HEIGHT]), [255,0,0], 2)
        cv2.circle(im,(a,start), 1, [0, 255, 0], 2)
        cv2.putText(im, f"Up = {up:+.4f} ({np.abs(up)/np.abs(u)*100:.1f}%)", (20,20), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im, f"Ui = {ui:+.4f} ({np.abs(ui)/np.abs(u)*100:.1f}%)", (20,20 + 10 + textSize), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im, f"Ud = {ud:+.4f} ({np.abs(ud)/np.abs(u)*100:.1f}%)", (20,20 + 20 + textSize * 2), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im, f"U = {u:+.4f}", (20,20 + 30 + textSize * 3), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im, f"V = {v:+.4f}", (20,20 + 40 + textSize * 4), font, fontScale, color, thickness, cv2.LINE_AA)
        
    GUI.showImage(im)
