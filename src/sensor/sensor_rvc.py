#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) RVBUST, Inc - All rights reserved.

from typing import Any, Tuple
import PyRVC as RVC

from enum import Enum
import numpy as np
import re
import copy
import sys
import json
import inspect

from src.sensor.frame import Frame
from src.sensor.sensor_base import SensorBase, SensorType, DeviceInfos

SensorRVCDevice = RVC.Device


class Sensor3DParam_RVC:
    def __init__(self) -> None:
        self.num_retry = 5

        self.sensor_name = "RVC"
        self.sn = None
        self.select_camera = RVC.CameraID_Left

        self.capture_options = None
        self.post_process_config = dict()

    def init_capture_options(self):
        if self.select_camera == RVC.CameraID_Both:
            self.capture_options = RVC.X2_CaptureOptions()
            self.capture_options.edge_noise_reduction_threshold = 0
        elif self.select_camera == RVC.CameraID_Left or self.select_camera == RVC.CameraID_Right:
            self.capture_options = RVC.X1_CaptureOptions()
        else:
            self.capture_options = None

    def from_dict(self, config: dict, prefix=""):
        self.num_retry = config.get(f"{prefix}num_retry", self.num_retry)
        self.sensor_name = config.get(f"{prefix}sensor_name", self.sensor_name)
        self.sn = config.get(f"{prefix}sn", self.sn)
        self.select_camera = RVC.CameraID.__members__[
            config.get(f"{prefix}select_camera", self.select_camera.name)]

        self.post_process_config = config.get(
            f"{prefix}post_process_config", self.post_process_config)

        self.set_capture_options(config.get(f"{prefix}capture_options", []))

    def to_dict(self):
        config = dict()
        config["num_retry"] = self.num_retry
        config["sensor_name"] = self.sensor_name
        config["sn"] = self.sn
        config["select_camera"] = self.select_camera.name

        # capture option
        if self.capture_options is not None:
            config["capture_options.capture_mode"] = self.capture_options.capture_mode.name
            config["capture_options.exposure_time_3d"] = self.capture_options.exposure_time_3d
            config["capture_options.exposure_time_2d"] = self.capture_options.exposure_time_2d
            config["capture_options.hdr_exposure_times"] = self.capture_options.hdr_exposure_times
            config["capture_options.light_contrast_threshold"] = self.capture_options.light_contrast_threshold

        # post process
        if "dbscan_removal" in self.post_process_config:
            config["post_process_config.dbscan_removal.distance_threshold_mm"] = self.post_process_config["dbscan_removal"]["distance_threshold_mm"]
            config["post_process_config.dbscan_removal.min_points"] = self.post_process_config["dbscan_removal"]["min_points"]
            config["post_process_config.dbscan_removal.min_num_cluster"] = self.post_process_config["dbscan_removal"]["min_num_cluster"]

        return config

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["capture_options"]
        return state

    def __setstate__(self, state):
        state["capture_options"] = None
        self.__dict__.update(state)

    def set_capture_options(self, config_capture_options):
        if self.capture_options is not None:
            for key in config_capture_options:
                if key == "roi":
                    continue
                value = config_capture_options[key]
                if key == "transform_to_camera":
                    if isinstance(self.capture_options, RVC.X2_CaptureOptions):
                        if value:
                            value = RVC.CameraID_Left
                        else:
                            value = RVC.CameraID_NONE
                if key == "smoothness":
                    value = RVC.SmoothnessLevel.__members__[value]
                if key == "capture_mode":
                    value = RVC.CaptureMode.__members__[value]
                if key == "projector_color":
                    value = RVC.ProjectorColor.__members__[value]
                if hasattr(self.capture_options, key):
                    setattr(self.capture_options, key, value)
                    continue

                res = re.match("^hdr_exposuretime_content_([1-3])$", key)
                if res is not None:
                    self.capture_options.SetHDRExposureTimeContent(
                        int(res[1]), value)
                res = re.match("^hdr_gain_3d_([1-3])$", key)
                if res is not None:
                    self.capture_options.SetHDRGainContent(
                        int(res[1]), int(value))
                res = re.match("^hdr_projector_brightness_([1-3])$", key)
                if res is not None:
                    self.capture_options.SetHDRProjectorBrightnessContent(
                        int(res[1]), value)
                res = re.match("^hdr_scan_times_([1-3])$", key)
                if res is not None:
                    self.capture_options.SetHDRScanTimesContent(
                        int(res[1]), value)

    def get_config_capture_options(self):
        capture_options = self.capture_options
        if capture_options is None:
            return {}
        config = {}
        variables_infos = inspect.getmembers(type(capture_options), lambda a: not(
            inspect.isroutine(a)) and type(a) == property)
        for variable_info in variables_infos:
            key = variable_info[0]
            if key in ["calc_normal", "calc_normal_radius", "downsample_distance", "filter_range", "noise_removal_distance", "noise_removal_point_number", "smoothness", "transform_to_camera"]:
                continue
            value = getattr(capture_options, key)
            value_type = type(value)
            if value_type in [int, float, bool]:
                config[key] = value
                continue
            if value_type == RVC.CameraID:
                config[key] = not(value != RVC.CameraID_NONE)
                continue
            # smoothness, capture_mode, projector_color
            if hasattr(value, "name"):
                config[key] = value.name
                continue
        if hasattr(capture_options, "GetHDRGainContent"):
            for i in range(1, 4):
                config[f"hdr_exposuretime_content_{i}"] = capture_options.GetHDRExposureTimeContent(
                    i)
        if hasattr(capture_options, "GetHDRGainContent"):
            for i in range(1, 4):
                config[f"hdr_gain_3d_{i}"] = capture_options.GetHDRGainContent(
                    i)
        if hasattr(capture_options, "GetHDRProjectorBrightnessContent"):
            for i in range(1, 4):
                config[f"hdr_projector_brightness_{i}"] = capture_options.GetHDRProjectorBrightnessContent(
                    i)
        if hasattr(capture_options, "GetHDRScanTimesContent"):
            for i in range(1, 4):
                config[f"hdr_scan_times_{i}"] = capture_options.GetHDRScanTimesContent(
                    i)
        return config


class Sensor3DStatus_RVC:
    def __init__(self) -> None:
        self.param = Sensor3DParam_RVC()
        self.support_x_mode = []

        self.frame = Frame()


class Sensor3D_RVC(SensorBase):
    def __init__(self) -> None:
        self.type = SensorType.RVC

        self.param = Sensor3DParam_RVC()

        self.status = Sensor3DStatus_RVC()

        self.dev = None
        self.sensor_rvc = None

    @staticmethod
    def list_device(device_type=RVC.SystemListDeviceTypeEnum.All) -> list:
        num_device, devices = RVC.SystemListDevices(device_type)
        return devices

    @staticmethod
    def find_device(sn=None, config=None) -> Tuple[bool, RVC.Device]:
        '''find correspondence device, or first device if sn is None

        Args:
            sn (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[bool, RVC.Device]: _description_
        '''
        if sn is None:
            num_device, devices = RVC.SystemListDevices(
                RVC.SystemListDeviceTypeEnum.All)
            if num_device > 0:
                device = devices[0]
            else:
                device = RVC.Device()
        else:
            device = RVC.SystemFindDevice(sn)
        return device.IsValid(), device

    @staticmethod
    def get_device_infos(dev: RVC.Device) -> Tuple[bool, DeviceInfos]:
        my_dev_infos = DeviceInfos()
        flag, dev_infos = dev.GetDeviceInfo()
        if not flag:
            return False, my_dev_infos
        my_dev_infos.sn = my_dev_infos.sensor_name = dev_infos.sn

        my_dev_infos.support_x_mode.clear()
        if int(RVC.CameraID_Left) & int(dev_infos.cameraid):
            my_dev_infos.support_x_mode.append(RVC.CameraID_Left)
        if int(RVC.CameraID_Right) & int(dev_infos.cameraid):
            my_dev_infos.support_x_mode.append(RVC.CameraID_Right)
        if dev_infos.support_x2:
            my_dev_infos.support_x_mode.append(RVC.CameraID_Both)

        return True, my_dev_infos

    def _update_intrinsic_param(self) -> bool:
        if self.sensor_rvc is None:
            return False
        if not self.sensor_rvc.IsOpen():
            return False
        if isinstance(self.sensor_rvc, RVC.X2):
            camera_id = RVC.CameraID_Left
            flag, intrinsic_matrix, distortion = self.sensor_rvc.GetIntrinsicParameters(
                camera_id)
            if not flag:
                return False
        else:
            flag, intrinsic_matrix, distortion = self.sensor_rvc.GetIntrinsicParameters()
            if not flag:
                return False
        intrinsic_matrix = np.array(intrinsic_matrix).reshape((3, 3))
        self.status.frame.intrinsic_matrix = intrinsic_matrix
        self.status.frame.distortion = np.array(distortion)
        return True

    def _update_extrinsic_matrix(self):
        if self.sensor_rvc is None:
            return False
        if not self.sensor_rvc.IsOpen():
            return False
        extrinsic_matrix = np.eye(4)
        if isinstance(self.sensor_rvc, RVC.X2):
            select_camera_id = self.param.capture_options.transform_to_camera
            camera_id = RVC.CameraID_Left
            if select_camera_id == RVC.CameraID_NONE:
                flag, extrinsic_matrix = self.sensor_rvc.GetExtrinsicMatrix(
                    camera_id)
                if not flag:
                    return False
        else:
            if not self.param.capture_options.transform_to_camera:
                flag, extrinsic_matrix = self.sensor_rvc.GetExtrinsicMatrix()
                if not flag:
                    return False
        extrinsic_matrix = np.array(extrinsic_matrix).reshape((4, 4))
        extrinsic_matrix[:3, 3] *= 1000
        self.status.frame.extrinsic_matrix = extrinsic_matrix
        return True

    def connect(self, dev: RVC.Device, **kwargs) -> bool:
        '''connect to device, kwargs can be sensor_config

        Args:
            dev (RVC.Device): _description_

        Returns:
            bool: _description_
        '''
        # update dev infos
        flag, dev_infos = dev.GetDeviceInfo()
        if not flag:
            return False
        self.status.support_x_mode.clear()
        if int(RVC.CameraID_Left) & int(dev_infos.cameraid):
            self.status.support_x_mode.append(RVC.CameraID_Left)
        if int(RVC.CameraID_Right) & int(dev_infos.cameraid):
            self.status.support_x_mode.append(RVC.CameraID_Right)
        if dev_infos.support_x2:
            self.status.support_x_mode.append(RVC.CameraID_Both)

        if len(self.status.support_x_mode) == 0:
            return False
        if "sensor_config" in kwargs:
            config_param = kwargs.get(
                "sensor_config", {}).get(f"{dev_infos.sn}", {})
        else:
            config_param = dict()
        select_camera = RVC.CameraID.__members__[
            config_param.get(f"select_camera", RVC.CameraID_NONE.name)]
        sensor_name = config_param.get("sensor_name", dev_infos.name)
        if select_camera == RVC.CameraID_NONE:
            select_camera = self.status.support_x_mode[0]
        elif select_camera not in self.status.support_x_mode:
            return False
        self.param.sn = dev_infos.sn
        self.param.select_camera = select_camera
        self.param.sensor_name = sensor_name

        self.status.param = copy.deepcopy(self.param)

        # connect to device
        self.disconnect()
        if self.param.select_camera == RVC.CameraID_Both:
            sensor_rvc = RVC.X2.Create(dev)
        else:
            sensor_rvc = RVC.X1.Create(dev, self.param.select_camera)
        if not sensor_rvc.IsValid():
            return False
        self.dev = dev
        self.sensor_rvc = sensor_rvc
        return True

    def is_connected(self):
        return self.sensor_rvc is not None

    def disconnect(self):
        if self.sensor_rvc is not None:
            if isinstance(self.sensor_rvc, RVC.X2):
                RVC.X2.Destroy(self.sensor_rvc)
            elif isinstance(self.sensor_rvc, RVC.X1):
                RVC.X1.Destroy(self.sensor_rvc)
            self.sensor_rvc = None
            self.dev = None
            self.status = Sensor3DStatus_RVC()

    def open(self, **kwargs) -> bool:
        '''kwargs can be sensor_config

        Returns:
            bool: _description_
        '''
        if self.sensor_rvc is None:
            return False
        if self.sensor_rvc.IsOpen():
            return True
        flag = self.sensor_rvc.Open()
        if not flag:
            self.close()
            return False
        self.param.init_capture_options()
        config_param = {}
        if "sensor_config" in kwargs:
            config_param = kwargs["sensor_config"].get(self.param.sn, {})
            self.status.frame.pose = np.array(
                config_param.get("frame_pose", np.eye(4)))
            self.param.from_dict(config_param)
        if "capture_options" not in config_param:
            ret, self.param.capture_options = self.sensor_rvc.LoadCaptureOptionParameters()

        self.status.param = copy.deepcopy(self.param)

        flag = self._update_intrinsic_param()
        if not flag:
            self.close()
        return flag

    def is_opened(self):
        if self.sensor_rvc is None:
            return False
        return self.sensor_rvc.IsOpen()

    def close(self):
        if self.sensor_rvc is not None:
            self.sensor_rvc.Close()

    def capture(self, **kwargs) -> Tuple[bool, Frame]:
        if self.sensor_rvc is None:
            return False, Frame()
        if "sensor_config" in kwargs:
            config_param = kwargs.get("sensor_config", {}).get(
                f"{self.param.sn}", {})
            self.param.set_capture_options(config_param.get("capture_options"))
        if self.status.param.capture_options is None or self.param.capture_options.transform_to_camera != self.status.param.capture_options.transform_to_camera:
            flag = self._update_extrinsic_matrix()
            if not flag:
                return False, Frame()
        self.status.frame.pm = None
        self.status.frame.image = None
        self.status.frame.mask = None
        for i in range(self.param.num_retry):
            flag = self.sensor_rvc.Capture(self.param.capture_options)
            if flag:
                break
            else:
                print(
                    f"sensor: {self.status.param.sensor_name} capture {i} frame failed")
        if not flag:
            print(f"sensor: {self.status.param.sensor_name} capture failed")
            return False, Frame()
        self.status.frame.pm = (
            np.array(self.sensor_rvc.GetPointMap()) * 1000).astype(np.float32)
        self.status.frame.mask = (
            ~np.isnan(self.status.frame.pm[:, :, 2])).astype(np.uint8)
        if isinstance(self.sensor_rvc, RVC.X2):
            image = np.array(self.sensor_rvc.GetImage(RVC.CameraID_Left))
        else:
            image = np.array(self.sensor_rvc.GetImage())
        self.status.frame.image = image
        self.status.param = copy.deepcopy(self.param)

        # post process
        self.post_process(self.status.frame, self.param.post_process_config)
        return True, self.status.frame

    def save_as_file_camera(self, sensor_config_path, file_camera_dir):
        with open(sensor_config_path, 'r') as f:
            config_sensor = json.load(f)
        sn = self.param.sn
        new_sn = f"file_{sn}"
        config_sensor[new_sn] = {}
        config = config_sensor[new_sn]
        config["type"] = SensorType.File.name
        config["sensor_name"] = self.param.sensor_name
        config["sn"] = new_sn
        config["post_process_config"] = self.param.post_process_config
        frame = self.status.frame
        config["frame_pose"] = frame.pose.tolist()
        config["intrinsic_matrix"] = frame.intrinsic_matrix.tolist()
        config["distortion"] = frame.distortion.tolist()
        config["extrinsic_matrix"] = frame.extrinsic_matrix.tolist()
        config["data_dir"] = f"{file_camera_dir}/{sn}/"
        config["unit"] = "mm"

        with open(sensor_config_path, 'w') as f:
            json.dump(config_sensor, f)

        frame.save(save_dir=config["data_dir"], id=-1)

    def save_config(self, sensor_config_path):
        with open(sensor_config_path, 'r') as f:
            config_sensor = json.load(f)
        sn = self.param.sn
        config_sensor[sn] = {}
        config = config_sensor[sn]
        config["type"] = self.type.name
        config["sn"] = sn
        config["sensor_name"] = self.param.sensor_name
        config["select_camera"] = self.param.select_camera.name
        config["frame_pose"] = self.status.frame.pose.tolist()
        config["post_process_config"] = self.param.post_process_config
        config["capture_options"] = self.param.get_config_capture_options()

        with open(sensor_config_path, 'w') as f:
            json.dump(config_sensor, f)


if __name__ == "__main__":
    from RVBUST import Vis
    import cv2
    import json
    from IPython import embed

    config_path = "./data/config_sensor.json"
    with open(config_path, "r") as f:
        config_sensor = json.load(f)

    RVC.SystemInit()
    sn = "P2GM453B001"

    sensor = Sensor3D_RVC()
    if sn in config_sensor:
        sensor.param.from_dict(config_sensor[sn])  # init X created param

    flag, dev = sensor.find_device(sensor.param.sn)
    if not flag:
        print("no device found")
        sys.exit()
    flag = sensor.connect(dev)
    if not flag:
        print("connect to device failed")
        sys.exit()
    flag = sensor.open()
    if not flag:
        print("open device failed")
        sys.exit()

    v = Vis.View()
    while True:
        flag, frame = sensor.capture(config_sensor=config_sensor)
        if not flag:
            print("capture failed")
            break
        v.Clear()
        v.Point(frame.pm.flatten(), 1, [0.5, 0.5, 0.5])
        v.Axes(np.eye(4).flatten(), 500, 3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", frame.image)
        key = chr(cv2.waitKey(30) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break

    sensor.disconnect()
    RVC.SystemShutdown()
    print("done")
