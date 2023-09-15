import PyRVC as RVC
import numpy as np
from IPython import embed
from src.sensor.sensor_factory import list_device, find_and_open_sensor
import json


class SensorManager:
    def __init__(self, config_sensor_file) -> None:
        RVC.SystemInit()
        self.device_list = []
        self.sensor_dict = dict()
        self.config_sensor = dict()
        self.rvc_list_device_type = RVC.SystemListDeviceTypeEnum.USB
        self.frame_dict = dict()

        if config_sensor_file is not None:
            self.update_sensor_config(config_sensor_file)

    def update_sensor_config(self, config_sensor_file):
        with open(config_sensor_file, 'r') as f:
            self.config_sensor = json.load(f)

    def update_param(self):
        for sn in self.sensor_dict:
            sensor = self.sensor_dict[sn]
            sensor.param.from_dict(self.config_sensor.get(sn, {}))

    def list_device(self, **kwargs) -> int:
        self.device_list = list_device(
            sensor_config=self.config_sensor, **kwargs)
        return len(self.device_list)

    def open(self, sn_list=[]) -> bool:
        for sn in sn_list:
            if sn in self.sensor_dict:
                continue
            flag, sensor = find_and_open_sensor(
                sn=sn, sensor_config=self.config_sensor, device_list=self.device_list)
            if not flag:
                print(f"open sensor: sn == {sn} failed")
                return False
            self.sensor_dict[sn] = sensor
        return True

    def close(self, sn_list=[]):
        for sn in sn_list:
            if sn in self.sensor_dict:
                sensor = self.sensor_dict[sn]
                sensor.close()
                sensor.disconnect()
                del self.sensor_dict[sn]

    def _is_all_sn_opened(self, sn_list=[]):
        if len(sn_list) == 0:
            return False
        for sn in sn_list:
            if sn not in self.sensor_dict:
                print(f"sensor sn == {sn} not opened")
                return False
        return True

    def capture(self, sn_list=[]) -> bool:
        self.frame_dict.clear()
        if not self._is_all_sn_opened(sn_list):
            return False
        for sn in sn_list:
            if sn not in self.sensor_dict:
                print(f"sensor sn == {sn} not connected")
                return False
        for sn in sn_list:
            sensor = self.sensor_dict[sn]
            flag, frame = sensor.capture()
            if not flag:
                print(f"sensor sn == {sn} capture failed")
                return False

        return True

    def get_total_point_cloud(self, sn_list=[], pose=np.eye(4)):
        total_pcd = np.array([], dtype=np.float32).reshape((-1, 3))

        if not self._is_all_sn_opened(sn_list):
            return False, total_pcd

        for sn in sn_list:
            sensor = self.sensor_dict[sn]
            frame = sensor.status.frame
            if frame.pm is None:
                print(
                    f"sensor sn == {sn} not capture 3d, please capture first")
                return False, total_pcd
            pcd = frame.get_transform_pm(T_b_w=pose)
            total_pcd = np.vstack([pcd, total_pcd])
        return True, total_pcd

    def __del__(self):
        for sn in self.sensor_dict:
            sensor = self.sensor_dict[sn]
            sensor.close()
            sensor.disconnect()
        # RVC.SystemShutdown()


if __name__ == "__main__":
    import sys
    from RVBUST import Vis
    import cv2

    sensor_manager = SensorManager("./data/config_sensor.json")

    sensor_manager.list_device()
    sn_list = ["P2GM453B001"]
    flag = sensor_manager.open(sn_list=sn_list)
    if not flag:
        sys.exit()
    v = Vis.View()
    while True:
        v.Clear()
        sensor_manager.update_param()
        flag = sensor_manager.capture(sn_list)
        if not flag:
            print(f"sn_list: {sn_list} capture failed")
            break
        flag, total_pcd = sensor_manager.get_total_point_cloud(sn_list)
        v.Point(total_pcd.flatten(), 1, [0.5, 0.5, 0.5])
        v.Axes(np.eye(4).flatten(), 500, 3)

        for sn in sn_list:
            sensor = sensor_manager.sensor_dict[sn]
            win_name = f"{sensor.param.sensor_name}_{sensor.param.sn}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, sensor.status.frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
        if key == "s":
            for sn in sn_list:
                sensor = sensor_manager.sensor_dict[sn]
                frame = sensor.status.frame
                frame.save(
                    f"./data/tmp/{sensor.param.sensor_name}_{sensor.param.sn}", id=-1)
    RVC.SystemShutdown()
    print("done")
