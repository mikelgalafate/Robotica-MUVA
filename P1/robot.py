import cv2
import numpy as np
from HAL import HAL
from GUI import GUI

# Enter sequential code!
k_p = 0
k_i = 0
k_d = 0

FILTRO = np.array([[],[]])

while True:
    # Enter iterative code!
    im = HAL.getImage()

    # Convertir a HSV

    # Aplicar rango

    # Escoger mayor blob

    GUI.showImage()