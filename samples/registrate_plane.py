import cv2
import numpy as np
import open3d as o3d
import sys
import os
from src.tools.common import get_common_parse


def segment_plane(pcd, distance_threshold_mm, num_iteration=30):
    if pcd.shape[0] < 100:
        return np.array([0, 0, 0]), np.array([0, 0, 1])
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    plane, inlier_idx = pcd_o3d.segment_plane(
        distance_threshold_mm, 3, num_iterations=num_iteration)
    inlier_points = pcd[inlier_idx]
    center = np.mean(inlier_points, axis=0)
    return np.array(center), np.array(plane[:3])


if __name__ == "__main__":
    import json
    from RVBUST import Vis
    import sys
    from src.tools.visualization import draw_plane
    from IPython import embed
    from src.sensor.sensor_manager import SensorManager

    parse = get_common_parse(
        "registrate sensor plane, use for crop_by_plane function")
    args = parse.parse_args()

    sn = args.sn
    config_sensor_path = args.config_sensor_path
    sensor_name = args.sensor_name

    sn_list = [sn]
    sensor_manager = SensorManager(config_sensor_path)

    config_sensor = sensor_manager.config_sensor
    # update crop_by_plane
    if sn not in config_sensor:
        config_sensor[sn] = {}
    if "post_process_config" not in config_sensor[sn]:
        config_sensor[sn]["post_process_config"] = {}
    if "crop_by_plane" not in config_sensor[sn]["post_process_config"]:
        config_sensor[sn]["post_process_config"]["crop_by_plane"] = {}
        config_sensor[sn]["post_process_config"]["crop_by_plane"]["crop_distance_range_mm"] = [
            0, 1000]
    config_sensor[sn]["post_process_config"]["crop_by_plane"]["enable"] = False
    config_sensor[sn]["sensor_name"] = sensor_name

    sensor_manager.list_device()
    flag = sensor_manager.open(sn_list=[sn])
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
            handle_src = v.Point(valid_pcd.flatten(), 1, [0.5, 0.5, 0.5])

            plane_center, plane_normal = segment_plane(valid_pcd, 10)
            if np.dot(plane_normals_direction, plane_normal) < 0:
                plane_normal *= -1

            crop_by_plane = config_sensor[sn]["post_process_config"]["crop_by_plane"]
            crop_by_plane["plane_center_mm"] = plane_center.tolist()
            crop_by_plane["plane_normal"] = plane_normal.tolist()

            handle_plane = draw_plane(
                v, plane_normal, plane_center, xLength=3000, yLength=3000)
            handle_plane_lower = draw_plane(
                v, plane_normal, plane_center + plane_normal * crop_by_plane["crop_distance_range_mm"][0], xLength=2000, yLength=2000, color=(0, 0, 1))
            handle_plane_upper = draw_plane(
                v, plane_normal, plane_center + plane_normal * crop_by_plane["crop_distance_range_mm"][1], xLength=2000, yLength=2000, color=(0, 0, 1))

            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
        if key == 's':
            config_sensor[sn]["post_process_config"]["crop_by_plane"]["enable"] = True
            if args.save_as_file_camera and hasattr(sensor, "save_as_file_camera"):
                sensor.save_as_file_camera(
                    config_sensor_path, args.file_camera_dir)
                print(
                    f"create file camera, raw data save in {args.file_camera_dir}/{sn}")
            if args.save_config and hasattr(sensor, "save_config"):
                sensor.save_config(args.config_sensor_path)
            else:
                sensor = sensor_manager.sensor_dict[sn]
                with open(config_sensor_path, 'w') as f:
                    json.dump(config_sensor, f)

            print(f"save config file to {config_sensor_path}")
            break
