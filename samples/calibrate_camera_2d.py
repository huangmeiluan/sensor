from typing import Tuple
import cv2
import numpy as np

from samples.utils.pose_detector import PoseDetector, CalibBoardType
from samples.utils.accuracy import compute_accuracy
from samples.utils.common import get_common_parse
from IPython import embed
from src.sensor.sensor_manager import SensorManager
import sys
from RVBUST import Vis


def calibrate_camera_2d(src_image_pts_list, src_obj_pts, image_shape) -> Tuple[bool, np.ndarray, np.ndarray]:
    image_pts_list = np.array(src_image_pts_list)
    image_pts_list = image_pts_list.reshape(
        (image_pts_list.shape[0], image_pts_list.shape[1], 1, image_pts_list.shape[2]))
    obj_pts = src_obj_pts.reshape(
        (1, src_obj_pts.shape[0], src_obj_pts.shape[1]))
    obj_pts_list = np.repeat(
        obj_pts, image_pts_list.shape[0], axis=0).astype(np.float32)
    ret, intrinsic_matrix, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list, image_pts_list, image_shape, None, None)
    return ret, intrinsic_matrix, dist_coeff


if __name__ == "__main__":

    parse = get_common_parse("registrate sensor pose, use for reconstruction")
    parse.add_argument("circle_center_distance_mm",
                       help="the distance of circle center", type=float)
    args = parse.parse_args()

    sn = args.sn
    config_sensor_path = args.config_sensor_path

    # pose detector
    pose_detector = PoseDetector()
    pose_detector.calib_board_type = CalibBoardType.RV_Undecoded_Circle
    pose_detector.pattern_per_rows = 4
    pose_detector.pattern_per_cols = 11
    pose_detector.circle_center_distance_mm = args.circle_center_distance_mm
    pose_detector.updata_obj_points()

    sn_list = [args.sn]

    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device()
    flag = sensor_manager.open(sn_list=[sn])
    if not flag:
        sys.exit()

    image_pts_list = []

    while True:
        flag = sensor_manager.capture(sn_list)
        if not flag:
            print("capture failed")
            sys.exit()

        sensor = sensor_manager.sensor_dict[sn]
        frame = sensor.status.frame

        image = frame.image
        ret, image_pts = pose_detector.find_image_points(image)
        if ret:
            image = pose_detector.draw(frame, image_pts)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        key = chr(cv2.waitKey(0) & 0xff)

        if ret and (key == 'a' or key == 's'):
            image_pts_list.append(image_pts)
            if args.save_as_file_camera and hasattr(sensor, "save_as_file_camera"):
                sensor.save_as_file_camera(
                    config_sensor_path, args.file_camera_dir)
        if key == 'q':
            break
        if key == 'e':
            embed()
        if key == 'c':
            ret, intrinsic_matrix, dist_coeff = calibrate_camera_2d(
                image_pts_list, pose_detector.obj_pts, image.shape[:2])
            print(f"ret: {ret}")
            print(f"intrinsic_matrix: {intrinsic_matrix}")
            print(f"dist_coeff: {dist_coeff}")

            if args.save_config and hasattr(sensor, "save_config"):
                sensor.status.frame.intrinsic_matrix = intrinsic_matrix
                sensor.status.frame.dist_coeff = dist_coeff
                sensor.save_config(args.config_sensor_path)
                print(f"save sensor config in {args.config_sensor_path}")
