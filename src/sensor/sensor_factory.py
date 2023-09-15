from csv import DictReader
from src.sensor.sensor_base import SensorType
from src.sensor.sensor_file3d import Sensor3D_File, Sensor3DParam_File
from src.sensor.sensor_rvc import Sensor3D_RVC, Sensor3DParam_RVC
from src.sensor.sensor_hik_2d import Sensor2D_Hik, DeviceType_Hik
import PyRVC as RVC
import numpy as np
from IPython import embed


def list_device(**kwargs):
    '''list device
    kwargs can be [rvc_list_device_type, sensor_config, list_sensor_type]
    to list file sensor, input sensor_config

    Returns:
        _type_: _description_
    '''
    list_sensor_type = 0xffff
    if "list_sensor_type" in kwargs:
        list_sensor_type = kwargs["list_sensor_type"]
    sensor_config = dict()
    if "sensor_config" in kwargs:
        sensor_config = kwargs["sensor_config"]

    device_list = []
    if list_sensor_type & SensorType.RVC == SensorType.RVC:
        rvc_list_device_type = RVC.SystemListDeviceTypeEnum.All
        if "rvc_list_device_type" in kwargs:
            rvc_list_device_type = kwargs["rvc_list_device_type"]
            if type(rvc_list_device_type) == str:
                rvc_list_device_type = RVC.SystemListDeviceTypeEnum.__members__[
                    rvc_list_device_type]
        rvc_device_list = Sensor3D_RVC.list_device(rvc_list_device_type)
        device_list = device_list + rvc_device_list
    if list_sensor_type & SensorType.File == SensorType.File:
        file_device_list = Sensor3D_File.list_device(sensor_config)
        device_list = device_list + file_device_list
    if list_sensor_type & SensorType.Hik_2D == SensorType.Hik_2D:
        hik_device_list = Sensor2D_Hik.list_device()
        device_list = device_list + hik_device_list
    return device_list


def get_device_infos(dev):
    if isinstance(dev, RVC.Device):
        flag, dev_infos = Sensor3D_RVC.get_device_infos(dev)
    elif isinstance(dev, Sensor3DParam_File):
        flag, dev_infos = Sensor3D_File.get_device_infos(dev)
    elif isinstance(dev, DeviceType_Hik):
        flag, dev_infos = Sensor2D_Hik.get_device_infos(dev)
    return flag, dev_infos


def find_sn(sensor_name, sensor_config):
    sn = None
    for key, value in sensor_config.items():
        if value.get("sensor_name", "") == sensor_name:
            sn = value["sn"]
            break
    return sn


def find_and_open_sensor(sn=None, sensor_name=None, **kwargs):
    '''find and open selected sensor, if sn and sensor_name is None, it means first sensor
    kwargs can be [device_list], it can speed up, or the key use to list device

    Args:
        sn (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    '''
    if "device_list" in kwargs:
        device_list = kwargs["device_list"]
    else:
        device_list = list_device(**kwargs)
    if len(device_list) == 0:
        return False, None
    if sn is None and sensor_name is not None:
        sn = find_sn(sensor_name=sensor_name,
                     sensor_config=kwargs.get("sensor_config", {}))
    if sn is None:
        found_dev = device_list[0]
    else:
        found_dev = None
        for dev in device_list:
            flag, dev_infos = get_device_infos(dev)
            if flag and sn == dev_infos.sn:
                found_dev = dev
                break
    if found_dev is None:
        return False, None
    if isinstance(found_dev, RVC.Device):
        sensor = Sensor3D_RVC()
    elif isinstance(found_dev, Sensor3DParam_File):
        sensor = Sensor3D_File()
    else:
        sensor = Sensor2D_Hik()
    flag = sensor.connect(
        found_dev, sensor_config=kwargs.get("sensor_config", {}))
    if not flag:
        return False, None
    flag = sensor.open(sensor_config=kwargs.get("sensor_config", {}))
    if not flag:
        return False, None
    return True, sensor


if __name__ == "__main__":
    import json
    from RVBUST import Vis
    import cv2
    import sys

    config_path = "./data/config_sensor.json"
    with open(config_path, 'r') as f:
        sensor_config = json.load(f)
    device_list = list_device(sensor_config=sensor_config,
                              rvc_list_device_type=RVC.SystemListDeviceTypeEnum.All, list_sensor_type=SensorType.Hik_2D)
    flag, sensor = find_and_open_sensor(
        sn="DA0235866", device_list=device_list, sensor_config=sensor_config)
    if not flag:
        print(f"open sensor failed")
        embed()
        sys.exit()

    v = Vis.View()
    while True:
        flag, frame = sensor.capture()
        if not flag:
            print("capture failed")
            break
        v.Clear()
        v.Point(frame.get_valid_points().flatten(), 1, [0.5, 0.5, 0.5])
        v.Axes(np.eye(4).flatten(), 500, 3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", frame.image)
        key = chr(cv2.waitKey(0) & 0xff)
        if key == 'e':
            embed()
        if key == 'q':
            break
    sensor.disconnect()
    RVC.SystemShutdown()
    print("done")
