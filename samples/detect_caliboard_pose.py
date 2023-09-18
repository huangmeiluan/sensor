import cv2
import numpy as np
import open3d as o3d
import copy

from samples.utils.pose_detector import PoseDetector, CalibBoardType
from samples.utils.accuracy import compute_accuracy
from samples.utils.common import get_common_parse


if __name__ == "__main__":
    import json
    from RVBUST import Vis
    import sys
    from samples.utils.visualization import draw_plane
    from IPython import embed
    from src.sensor.sensor_manager import SensorManager

    parse = get_common_parse("registrate sensor pose, use for reconstruction")
    # parse.add_argument("--use_as_camera_base_pose",
    #                    help="set current pose as camera's base pose", action="store_true")
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
            if ret:
                image = pose_detector.draw(
                    frame=frame, image_pts=image_pts, pose=obj_pose)
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
            else:
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    image = copy.deepcopy(image)
                for pt in image_pts:
                    cv2.circle(image, np.int0(pt), 5, (0, 0, 255), 2)
                print(
                    f"detect caliboard pose of sn == {sensor.param.sn} failed")

            valid_pcd = frame.get_transform_pm(
                T_b_w=new_pose @ np.linalg.inv(frame.pose))
            handle_src = v.Point(valid_pcd.flatten(), 1, [0.5, 0.5, 0.5])

            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':  # embed
            embed()
        if key == 'q':  # quit
            break
        if key == 'u':  # use as camera base pose
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                frame = sensor.status.frame
                frame.pose = frame_pose_dict[sn]
                if args.save_config and hasattr(sensor, "save_config"):
                    sensor.save_config(args.config_sensor_path)
                else:
                    config_sn = sensor_manager.config_sensor["sn"]
                    config_sn["frame_pose"] = frame.pose
                    with open(config_sensor_path, 'w') as f:
                        json.dump(sensor_manager.config_sensor, f)
        if key == 's':  # save
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                frame = sensor.status.frame
                frame.pose = frame_pose_dict[sn]
                if args.save_as_file_camera and hasattr(sensor, "save_as_file_camera"):
                    sensor.save_as_file_camera(
                        config_sensor_path, args.file_camera_dir)
                    print(
                        f"create file camera, raw data save in {args.file_camera_dir}/{sn}")
                if args.save_config and hasattr(sensor, "save_config"):
                    sensor.save_config(args.config_sensor_path)
