#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

import numpy as np
from typing import Tuple
import cv2
import glob
import sys
import re

from src.sensor.sensor_base import DeviceInfos, SensorBase, SensorType
from src.sensor.frame import Frame
from IPython import embed
import PyRVC as RVC


class Sensor3DParam_File:
    def __init__(self) -> None:
        self.data_dir = "./"
        self.image_subfix = ".png"
        self.pc_subfix = ".ply"

        self.unit_scale = 1

        self.select_camera = RVC.CameraID_Left

        self.sn = 'file'
        self.sensor_name = ""
        self.post_process_config = dict()

    def from_dict(self, config: dict):
        self.data_dir = config.get("data_dir", self.data_dir)
        self.image_subfix = config.get("image_subfix", self.image_subfix)
        self.pc_subfix = config.get("pc_subfix", self.pc_subfix)

        self.sn = config.get("sn", self.sn)
        self.sensor_name = config.get("sensor_name", self.sensor_name)

        unit = config.get("unit", None)
        if unit == 'm':
            self.unit_scale = 1000
        self.post_process_config = config.get(
            "post_process_config", self.post_process_config)


class Sensor3DStatus_File:
    def __init__(self) -> None:
        self.frame = Frame()


class Sensor3D_File(SensorBase):
    def __init__(self) -> None:
        self.type = SensorType.File

        self.param = Sensor3DParam_File()
        self.status = Sensor3DStatus_File()

        self._file_names = None

    @staticmethod
    def list_device(config_sensor: dict):
        param_list = []
        for sensor_name in config_sensor:
            sensor_infos = config_sensor[sensor_name]
            # print(f"sensor_infos: {sensor_infos}")
            if sensor_infos.get("type", "") == "File":
                param = Sensor3DParam_File()
                param.from_dict(sensor_infos)
                param_list.append(param)
        return param_list

    @staticmethod
    def find_device(sn, config_sensor: dict) -> Tuple[bool, Sensor3DParam_File]:
        param_list = Sensor3D_File.list_device(config_sensor)
        for param in param_list:
            if sn == param.sn:
                return True, param
        return False, Sensor3DParam_File()

    @staticmethod
    def get_device_infos(dev) -> Tuple[bool, DeviceInfos]:
        my_device_infos = DeviceInfos()
        my_device_infos.sn = dev.sn
        my_device_infos.sensor_name = dev.sensor_name
        my_device_infos.support_x_mode = [dev.select_camera]
        return True, my_device_infos

    def _get_file_names(self):
        while True:
            abs_image_files = glob.glob(
                f"{self.param.data_dir}/**/*{self.param.image_subfix}", recursive=True)
            abs_pc_files = glob.glob(
                f"{self.param.data_dir}/**/*{self.param.pc_subfix}", recursive=True)
            image_files = [name[len(
                self.param.data_dir):-len(self.param.image_subfix)] for name in abs_image_files]
            pc_files = [name[len(
                self.param.data_dir):-len(self.param.image_subfix)] for name in abs_pc_files]
            file_names = list(set(pc_files).intersection(set(image_files)))
            try:
                file_names.sort(key=lambda name: int(
                    re.findall("\d+", name)[0]))
            except:
                file_names.sort()
            assert len(
                file_names) > 0, f"no file found in {self.param.data_dir}"
            for file_name in file_names:
                yield file_name

    def _update_frame_info(self, config):
        frame = self.status.frame
        frame.intrinsic_matrix = np.array(config.get(
            "intrinsic_matrix", frame.intrinsic_matrix))
        frame.distortion = np.array(config.get("distortion", frame.distortion))
        frame.extrinsic_matrix = np.array(config.get(
            "extrinsic_matrix", frame.extrinsic_matrix))
        frame.pose = np.array(config.get("frame_pose", frame.pose))

    def connect(self, dev, **kwargs) -> bool:
        self.param = dev
        self._update_frame_info(kwargs.get(self.param.sn, {}))
        return True

    def open(self, **kwargs) -> bool:
        self._update_frame_info(kwargs.get(
            "sensor_config", {}).get(self.param.sn, {}))

        self._file_names = self._get_file_names()
        return True

    def capture(self, **kwargs):
        if self._file_names is None:
            return False, Frame()
        file_name = next(self._file_names)
        ret = self.status.frame.load_from_file(pm_path=f"{self.param.data_dir}/{file_name}{self.param.pc_subfix}",
                                               image_path=f"{self.param.data_dir}/{file_name}{self.param.image_subfix}",
                                               unit_scale=self.param.unit_scale)
        print(f"load file {self.param.data_dir}/{file_name}")
        if not ret:
            return ret, Frame()
        self.post_process(self.status.frame, self.param.post_process_config)
        return ret, self.status.frame


if __name__ == "__main__":
    from RVBUST import Vis
    import json
    from IPython import embed

    config_path = "./data/config_sensor.json"
    with open(config_path, "r") as f:
        config_sensor = json.load(f)

    sensor = Sensor3D_File()
    sensor.param.from_dict(config_sensor["file_P2GM453B001"])

    flag = sensor.open()

    v = Vis.View()
    while True:
        v.Clear()
        flag, frame = sensor.capture()
        if not flag:
            print("capture failed")
            sys.exit()
        v.Point(frame.get_valid_points().flatten(), 1, [0.5, 0.5, 0.5])
        v.Axes(np.eye(4).flatten(), 500, 3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
