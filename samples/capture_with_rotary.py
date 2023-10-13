from turtle import pos
import cv2
import numpy as np

from src.tools.common import get_common_parse

from RVBUST import Vis
import sys
from IPython import embed
from src.sensor.sensor_manager import SensorManager
from src.rotary.rotary_platform_controller import RPController
import PyRVC as RVC

if __name__ == "__main__":
    parser = get_common_parse(
        "capture. \
        focus on image window and press: \
        'e': embed, \
        'q': quit, \
        's': save config, \
        'f': (save as) file camera")
    parser.add_argument("--rotate_angle_deg", help="rotate angle per step in degrees, default = %(default)s", default=5, type=float)
    parser.add_argument("--rotate_speed_deg_per_second", help="rotate speed degrees per second, default = %(default)s", default=5, type=float)
    args = parser.parse_args()

    config_sensor_path = args.config_sensor_path

    sn_list = args.sn_list

    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device(
        list_sensor_type=args.list_sensor_type, rvc_list_device_type=args.rvc_list_device_type)
    flag = sensor_manager.open(sn_list=sn_list)
    if not flag:
        sys.exit()
    
    # rotary
    rotary = RPController(ip_addr="10.10.10.10")
    rotary.connect()
    rotary.set_step_speed(args.rotate_speed_deg_per_second)
    embed()


    v = Vis.View()
    while True:
        pose_rotary = rotary.get_pose()
        pose = np.linalg.inv(pose_rotary)

        flag = sensor_manager.capture(sn_list)
        if not flag:
            print("capture failed")
            sys.exit()

        v.Axes(np.eye(4).flatten(), 1000, 1)

        ret, total_pcd = sensor_manager.get_total_point_cloud(sn_list=sn_list, pose=pose)

        v.Point(total_pcd.flatten(), 1, [.5, .5, .5])

        for sn in sn_list:
            sensor = sensor_manager.sensor_dict[sn]
            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, sensor.status.frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
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


        rotary.step_move(args.rotate_angle_deg)
        