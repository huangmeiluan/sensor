import cv2
import numpy as np
import open3d as o3d
import copy

from src.tools.pose_detector import PoseDetector, CalibBoardType
from src.tools.accuracy import compute_accuracy
from src.tools.common import get_common_parse


if __name__ == "__main__":
    import json
    from RVBUST import Vis
    import sys
    from src.tools.visualization import draw_plane
    from IPython import embed
    from src.sensor.sensor_manager import SensorManager

    parse = get_common_parse(
        "registrate caliboard pose. \
        focus on image window and press: \
        'e': embed, \
        'q': quit, \
        's': save config, \
        'f': (save as) file camera, \
        'b': (set as) base pose")
    parse.add_argument("circle_center_distance_mm",
                       help="the distance of circle center", type=float)
    args = parse.parse_args()

    config_sensor_path = args.config_sensor_path

    # pose detector
    pose_detector = PoseDetector()
    pose_detector.calib_board_type = CalibBoardType.RV_Undecoded_Circle
    pose_detector.pattern_per_rows = 4
    pose_detector.pattern_per_cols = 11
    pose_detector.circle_center_distance_mm = args.circle_center_distance_mm
    pose_detector.updata_obj_points()

    sn_list = args.sn_list

    sensor_manager = SensorManager(config_sensor_path)
    sensor_manager.list_device()
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
        frame_pose_dict = dict()
        for sn in sn_list:
            sensor = sensor_manager.sensor_dict[sn]
            frame = sensor.status.frame

            ret, obj_pose, image_pts, obj_pts = pose_detector.compute_obj_pose(
                frame, use_pnp=frame.pm is None)
            new_pose = np.linalg.inv(obj_pose)
            frame_pose_dict[sn] = new_pose
            image = frame.image
            image = pose_detector.draw(
                frame=frame, image_pts=image_pts, pose=obj_pose, pattern_was_found=ret)
            if ret:
                if obj_pts.size > 0:
                    print(
                        f"the accuracy of {sensor.param.sensor_name}_{sensor.param.sn}:")
                    compute_accuracy(obj_pose, obj_pts,
                                     frame.extrinsic_matrix, pose_detector.obj_pts)

                    T_pcd_2_base = new_pose @ frame.extrinsic_matrix
                    trans_obj_points = np.dot(
                        T_pcd_2_base[:3, :3], obj_pts.T).T + T_pcd_2_base[:3, 3]
                    v.Point(trans_obj_points.flatten(), 5, np.random.rand(3))

                    print(
                        f"caliboard pose in camera: xyz (mm) = {obj_pose[:3, 3]}")
                    print(
                        f"caliboard pose in camera base: xyz (mm) = {obj_pose[:3, 3]}")
                    print("save pose in calib_pose_in_camera.txt")
                    np.savetxt("calib_pose_in_camera.txt", obj_pose)
                    np.savetxt("calib_pose_in_camera_base.txt",
                               frame.pose @ obj_pose)

            valid_pcd = frame.get_transform_pm(
                T_b_w=new_pose @ np.linalg.inv(frame.pose))
            handle_src = v.Point(valid_pcd.flatten(), 1, [0.5, 0.5, 0.5])

            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':  # embed
            embed()
        if key == 'q':  # quite
            break
        if key == 'b':  # base pose
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                frame = sensor.status.frame
                frame.pose = frame_pose_dict[sn]
                print("set sensor: {sn} base pose")
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
