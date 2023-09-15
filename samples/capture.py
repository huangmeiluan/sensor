import cv2
import numpy as np

from samples.utils.common import get_common_parse

from RVBUST import Vis
import sys
from IPython import embed
from src.sensor.sensor_manager import SensorManager
import PyRVC as RVC

if __name__ == "__main__":
    parse = get_common_parse("capture")
    args = parse.parse_args()

    sn = args.sn
    config_sensor_path = args.config_sensor_path

    sn_list = [args.sn]

    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device(
        list_sensor_type=args.list_sensor_type, rvc_list_device_type=args.rvc_list_device_type)
    flag = sensor_manager.open(sn_list=sn_list)
    if not flag:
        sys.exit()

    v = Vis.View()
    while True:
        v.Clear()
        flag = sensor_manager.capture(sn_list)
        if not flag:
            print("capture failed")
            sys.exit()

        v.Axes(np.eye(4).flatten(), 1000, 1)

        ret, total_pcd = sensor_manager.get_total_point_cloud(sn_list=sn_list)

        v.Point(total_pcd.flatten(), 1, [.5, .5, .5])

        sensor = sensor_manager.sensor_dict[sn]
        win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, sensor.status.frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
        if key == 's':
            sensor = sensor_manager.sensor_dict[sn]
            if args.save_as_file_camera and hasattr(sensor, "save_as_file_camera"):
                sensor.save_as_file_camera(
                    config_sensor_path, args.file_camera_dir)
                print(
                    f"create file camera, raw data save in {args.file_camera_dir}/{sn}")
            if args.save_config and hasattr(sensor, "save_config"):
                sensor.save_config(args.config_sensor_path)
                print(f"save sensor config in {args.config_sensor_path}")
