import cv2
from tqdm import tqdm
import numpy as np


def get_chessboard_points(chessboard_shape, dx, dy):

    return [[round(i * dx, 1), round(j * dy, 1), 0]
            for i in range(chessboard_shape[1])
            for j in range(chessboard_shape[0])]

def default_camera_calibration_images():
    print("Images are required for default camera calibration")
    print("\tPlease, place the OpenCV 9x6 Chessboard in the camera")
    print("\t\tChessboard must be in a 25mm/side size")
    print("\tMove the Chessboard around the camera and press C to take a picture, Q to exit program.")
    print("\t12 pictures are needed.")
    print("#######################################################################################")
    print("####################################### WARNING #######################################")
    print("########## Make sure the whole chessboard is visible for a good calibration ###########")
    print("#######################################################################################")
    print("#######################################################################################")
    calib_images = []
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and len(calib_images) < 12:
        ret, frame = cap.read()
        cv2.imshow("Camera calibration", frame)
        wk = cv2.waitKey(1)
        if wk & 0xFF == ord('c'):
            calib_images.append(frame)
        elif wk & 0xFF == ord('q'):
            cap.release()
            exit()
        print(f"{len(calib_images)}/12 images were taken", end="\r")
    cap.release()
    cv2.destroyAllWindows()
    return calib_images


def calibrate_camera(calib_images):
    print("\n\tFinding calibration points")
    corners = [cv2.findChessboardCorners(im, patternSize=(9, 6)) for im in tqdm(calib_images)]
    cb_points = np.asarray([get_chessboard_points((9, 6), 25, 25)], dtype=np.float32)
    valid_corners = [corner[1] for corner in corners if corner[0]]
    num_valid_images = len(valid_corners)
    if num_valid_images == 0:
        return None, 0
    else:
        object_points = np.asarray([cb_points for _ in range(num_valid_images)], dtype=np.float32)
        image_points = np.asarray(valid_corners, dtype=np.float32)
        rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                                         image_points,
                                                                         calib_images[0].shape[0:2],
                                                                         None,
                                                                         None,
                                                                         flags=cv2.CALIB_FIX_ASPECT_RATIO)
        camera_values = (rms, intrinsics, dist_coeffs, rvecs, tvecs)
        return camera_values, num_valid_images
