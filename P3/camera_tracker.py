import cv2
import glob
import utils
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
matplotlib.use("TkAgg")


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Path for the ", required=False)
parser.add_argument("-c", "--calibration", help="Path for the calibration images directory", required=False)
args = parser.parse_args()

print("Camera calibration:")
if args.video is None or args.calibration is None:
    if (args.video is None) ^ (args.calibration is None):
        print("Either both or none of the arguments must be specified")
        print("Would you like to continue with the default camera? [y/n]: ", end="")
        ans = input()
        if ans != 'y' and ans != 'Y':
            exit()
    print()
    args.video = 0
    calib_images = utils.default_camera_calibration_images()
    camera_values, num_valid_images = utils.calibrate_camera(calib_images)

else:
    print("\tReading calibration images")
    calib_images = [cv2.resize(cv2.imread(im), (1280, 720)) for im in tqdm(glob.glob(args.calibration + "/*"))]
    camera_values, num_valid_images = utils.calibrate_camera(calib_images)

while num_valid_images == 0:
    print("No valid images were found")
    print("Would you like to calibrate again with the default camera? [y/n]: ", end="")
    ans = input()
    if ans == 'y' or ans == 'Y':
        calib_images = utils.default_camera_calibration_images()
        camera_values, num_valid_images = utils.calibrate_camera(calib_images)
    elif ans == 'n' or ans == 'N':
        exit()

rms, intrinsics, dist_coeffs, rvecs, tvecs = camera_values

print("\nCalibration complete")
print("Corners standard intrinsics:\n", intrinsics)
print("Corners standard dist_coefs:\n", dist_coeffs)
print("rms:", rms)

sr = 50
object_points = np.array([[-sr, sr, 0],
                          [sr, sr, 0],
                          [sr, -sr, 0],
                          [-sr, -sr, 0]]).astype(np.float32)

plt.figure()
axes = plt.axes(projection="3d")

centers = []

cap = cv2.VideoCapture(args.video)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_params)
        aruco_corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dictionary, parameters=aruco_params)
        if aruco_corners:
            for i, tag_id in enumerate(ids):
                print(f"Tag num.:{tag_id[0]} found", end="\r")
                for corner in aruco_corners[i].squeeze().astype("int"):
                    cv2.circle(frame, corner, 2, (0, 255, 0), 5)
                tag_corners = np.hstack((aruco_corners[i].squeeze(), np.zeros((4, 1)))).astype(np.float32)
                tag_corners = aruco_corners[i].squeeze().astype(np.float32)
                _, rotation, translation = cv2.solvePnP(object_points,
                                                        tag_corners,
                                                        intrinsics,
                                                        dist_coeffs)
                frame = cv2.drawFrameAxes(frame, intrinsics, dist_coeffs, rotation, translation, 80, thickness=2)

            R, _ = cv2.Rodrigues(rotation)
            C = -R.T @ translation
            centers.append(C)

            axes.cla()
            axes.plot(np.array(centers)[:, 0], np.array(centers)[:, 1], np.array(centers)[:, 2], "k")
            axes.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2])
            axes.set_xlim(-300, 300)
            axes.set_ylim(-300, 300)
            axes.set_zlim(0, 500)
            axes.set_xlabel("Eje X")
            axes.set_ylabel("Eje Y")
            axes.set_zlabel("Eje Z")
            plt.draw()
            plt.pause(0.01)

        else:
            print("No ArUco tag found", end="\r")
        cv2.imshow("Video", frame)
        wk = cv2.waitKey(16)
        if wk & 0xFF == ord('p'):
            wk = cv2.waitKey(0)
        if wk & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            plt.close()
    else:
        cap.release()

print("\n\nEnd of detection")
plt.close()
cv2.destroyAllWindows()
