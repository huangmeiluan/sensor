from src.sensor.sensor_manager import SensorManager
from src.tools.hand_eye_calibrate import HandEyeCalibrateManager
from src.tools.common import get_common_parse
from src.rotary.rotary_platform_controller import RPController
import numpy as np
import cv2
import sys
from IPython import embed
from RVBUST import Vis
import time

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
    parse.add_argument("--rotate_angle_deg",
                       help="rotate angle per step in degrees, default = %(default)s", default=5, type=float)
    parse.add_argument("--rotate_speed_deg_per_second",
                       help="rotate speed degrees per second, default = %(default)s", default=5, type=float)
    args = parse.parse_args()

    sn_list = args.sn_list
    config_sensor_path = args.config_sensor_path
    circle_center_distance_mm = args.circle_center_distance_mm
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
    hand_eye_calibrate_manager = HandEyeCalibrateManager()
    hand_eye_calibrate_manager.set_circle_center_distance(
        circle_center_distance_mm=circle_center_distance_mm)

    # rotary manager
    rotary = RPController(ip_addr="10.10.10.10")
    rotary.connect()
    rotary.set_step_speed(args.rotate_speed_deg_per_second)
    embed()

    v = Vis.View()
    while True:
        v.Clear()
        # move and get pose
        pose_rotary = rotary.get_pose()

        # capture
        flag, frame = sensor.capture()
        if not flag:
            print("capture failed")
            sys.exit()

        ret, T_calib_2_cam, points_2d, points_3d = hand_eye_calibrate_manager.detect_calib_pose(
            frame=frame, pose_in_cam=True)

        debug_image = hand_eye_calibrate_manager.draw_points(
            frame=frame, pose=T_calib_2_cam, pattern_was_found=ret, image_pts=points_2d)

        v.Point(frame.get_valid_points(), 1, [.5, .5, .5])
        win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, debug_image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'a':  # append
            ret, position_infos = hand_eye_calibrate_manager.add_data(
                frame=frame, pose_robot_end=pose_rotary)
            if ret:
                print(
                    f"add data success")
            else:
                print(f"add data failed")
            print(
                f"total data: {len(hand_eye_calibrate_manager.position_infos_list)}")
        if key == 'c':  # compute
            ret, T_cam_2_rotary = hand_eye_calibrate_manager.calibrate_rotary()
            if ret:
                frame.pose = T_cam_2_rotary
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
        rotary.step_move(angle_in_deg=args.rotate_angle_deg)
        time.sleep(0.5)
