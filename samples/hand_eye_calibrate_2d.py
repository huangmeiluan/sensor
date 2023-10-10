from src.sensor.sensor_manager import SensorManager
from src.tools.hand_eye_calibrate import HandEyeCalibrateManager_2d, PositionInfos_2d
from src.tools.common import get_common_parse
import numpy as np
import cv2
import sys
from IPython import embed
from RVBUST import Vis

if __name__ == "__main__":
    parse = get_common_parse(
        "hand eye calibrate 2d. \
        focus on image window and press: \
        'a': append (add data)\
        'c': compute \
        'e': embed, \
        'q': quit, \
        's': save config, \
        'f': (save as) file camera")
    parse.add_argument("circle_center_distance_mm",
                       help="the distance of circle center", type=float)
    parse.add_argument("eye_in_hand", type=bool, default=True)
    args = parse.parse_args()

    sn_list = args.sn_list
    config_sensor_path = args.config_sensor_path
    circle_center_distance_mm = args.circle_center_distance_mm
    eye_in_hand = args.eye_in_hand
    sn = sn_list[0]

    # sensor manager
    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device(
        list_sensor_type=args.list_sensor_type, rvc_list_device_type=args.rvc_list_device_type)
    flag = sensor_manager.open(sn_list=sn_list)
    if not flag:
        sys.exit()

    sensor = sensor_manager.sensor_dict[sn]

    # hand eye calibrate manager
    hand_eye_calibrate_manager = HandEyeCalibrateManager_2d()
    hand_eye_calibrate_manager.set_circle_center_distance(
        circle_center_distance_mm=circle_center_distance_mm)

    # robot manager

    v = Vis.View()
    while True:
        v.Clear()
        # move and get pose
        pose_robot_end = np.eye(4)

        # capture
        flag, frame = sensor.capture()
        if not flag:
            print("capture failed")
            sys.exit()
        ret, image_pts = hand_eye_calibrate_manager.detect_calib_points(
            frame.image)
        debug_image = hand_eye_calibrate_manager.draw_points(
            frame.image, ret, image_pts=image_pts)

        win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, debug_image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'a':  # append
            ret, position_infos = hand_eye_calibrate_manager.add_data(
                frame.image, pose_robot_end)
            if ret:
                print(
                    f"add data success")
            else:
                print(f"add data failed")
            print(
                f"total data: {len(hand_eye_calibrate_manager.position_infos_list)}")
        if key == 'c':  # compute
            ret, intrinsic_matrix, dist_coeff_rv, hand_eye_result = hand_eye_calibrate_manager.calibrate(
                eye_in_hand, calib_sensor=True)
            if ret:
                frame.intrinsic_matrix = intrinsic_matrix
                frame.dist_coeff_rv = dist_coeff_rv
                frame.pose = hand_eye_result
            else:
                print(f"calibrate failed, embed here")
                embed()
        if key == 'e':  # embed
            embed()
        if key == 'q':  # quite
            break
        if key == 's':  # save config
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                if hasattr(sensor, "save_config"):
                    sensor.save_config(args.config_sensor_path)
                    print(f"save sensor config in {args.config_sensor_path}")
                else:
                    print(f"sensor: {sn} has no 'save_config' function")
        if key == 'f':  # file camera
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                if hasattr(sensor, "save_as_file_camera"):
                    sensor.save_as_file_camera(
                        config_sensor_path, args.file_camera_dir)
                    print(
                        f"create file camera, raw data save in {args.file_camera_dir}/{sn}")
                else:
                    print(
                        f"sensor: {sn} has no 'save_as_file_camera' function")
