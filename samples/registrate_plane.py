import cv2
import numpy as np
import open3d as o3d
import sys
import os
from src.tools.common import get_common_parse


if __name__ == "__main__":
    import json
    from RVBUST import Vis
    import sys
    from src.tools.visualization import draw_plane
    from IPython import embed
    from src.sensor.sensor_manager import SensorManager

    parser = get_common_parse(
        "registrate sensor plane, use for crop_by_plane function \
        focus on image window and press: \
        'e': embed, \
        'q': quit, \
        's': save config, \
        'f': (save as) file camera")
    parser.add_argument("--distance_threshold",
                        help="default = %(default)s", type=float, default=3)
    args = parser.parse_args()

    config_sensor_path = args.config_sensor_path

    sn_list = args.sn_list
    sensor_manager = SensorManager(config_sensor_path)

    config_sensor = sensor_manager.config_sensor
    # update crop_by_plane
    for sn in sn_list:
        if sn not in config_sensor:
            config_sensor[sn] = {}
        if "post_process_config" not in config_sensor[sn]:
            config_sensor[sn]["post_process_config"] = {}
        if "crop_by_plane" not in config_sensor[sn]["post_process_config"]:
            config_sensor[sn]["post_process_config"]["crop_by_plane"] = {}
            config_sensor[sn]["post_process_config"]["crop_by_plane"]["crop_distance_range_mm"] = [
                5, 1000]
        config_sensor[sn]["post_process_config"]["crop_by_plane"]["enable"] = False

    sensor_manager.list_device(
        list_sensor_type=args.list_sensor_type, rvc_list_device_type=args.rvc_list_device_type)
    flag = sensor_manager.open(sn_list=sn_list)
    if not flag:
        sys.exit()

    plane_normals_direction = np.array([0, 0, 1])

    v = Vis.View()
    while True:
        v.Clear()
        flag = sensor_manager.capture(sn_list)
        if not flag:
            print("capture failed")
            sys.exit()

        v.Axes(np.eye(4).flatten(), 1000, 1)
        for sn in sn_list:
            sensor = sensor_manager.sensor_dict[sn]
            frame = sensor.status.frame

            valid_pcd = frame.get_valid_points()

            plane_center, plane_normal, inlier_indices = cv2.fit_plane_ransac(
                valid_pcd, distance_threshold=args.distance_threshold)
            plane_center = np.array(plane_center)
            plane_normal = np.array(plane_normal)
            if np.dot(plane_normals_direction, plane_normal) < 0:
                plane_normal *= -1

            crop_by_plane = config_sensor[sn]["post_process_config"]["crop_by_plane"]
            crop_by_plane["plane_center_mm"] = plane_center.tolist()
            crop_by_plane["plane_normal"] = plane_normal.tolist()

            handle_plane = draw_plane(
                v, plane_normal, plane_center, xLength=500, yLength=500)
            handle_plane_lower = draw_plane(
                v, plane_normal, plane_center + plane_normal * crop_by_plane["crop_distance_range_mm"][0], xLength=2000, yLength=2000, color=(0, 0, 1))
            handle_plane_upper = draw_plane(
                v, plane_normal, plane_center + plane_normal * crop_by_plane["crop_distance_range_mm"][1], xLength=2000, yLength=2000, color=(0, 0, 1))

            v.Point(valid_pcd.flatten(), 1, [0.5, 0.5, 0.5])
            v.Point(valid_pcd[inlier_indices], 2, [1, 0, 0])
            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':  # embed
            embed()
        if key == 'q':  # quite
            break
        if key == 's':  # save config
            for sn in sn_list:
                config_sensor[sn]["post_process_config"]["crop_by_plane"]["enable"] = True
                sensor = sensor_manager.sensor_dict[sn]
                if hasattr(sensor, "save_config"):
                    sensor.save_config(args.config_sensor_path)
                    print(f"save sensor config in {args.config_sensor_path}")
                else:
                    print(
                        f"sensor: {sn} has no 'save_config' function, so update to {config_sensor_path}")
                    with open(config_sensor_path, 'w') as f:
                        json.dump(config_sensor, f)
                config_sensor[sn]["post_process_config"]["crop_by_plane"]["enable"] = False
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
