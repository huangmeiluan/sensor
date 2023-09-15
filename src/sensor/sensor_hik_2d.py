from os import stat
import sys  # noqa
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")  # noqa
import numpy as np
from typing import Tuple
from src.sensor.sensor_base import SensorBase, SensorType, DeviceInfos
from src.sensor.frame import Frame
from MvCameraControl_class import *
import json

DeviceType_Hik = MV_CC_DEVICE_INFO


class HikConfigName:
    PayloadSize = "PayloadSize"
    Timeout_ms = "timeout"
    GevSCPSPacketSize = "GevSCPSPacketSize"
    TriggerMode = "TriggerMode"
    BufferSize = "BufferSize"
    SN = "sn"

    ExposureTime_us = "ExposureTime"


class Sensor2DParam_Hik:
    def __init__(self) -> None:
        self.sn = ""
        self.sensor_name = "hik"


class Sensor2DStatus_Hik:
    def __init__(self) -> None:
        self.frame = Frame()


class Sensor2D_Hik(SensorBase):
    def __init__(self) -> None:
        super().__init__()

        self.type = SensorType.Hik_2D
        self.param = Sensor2DParam_Hik()
        self.status = Sensor2DStatus_Hik()

        self._device_info = MV_CC_DEVICE_INFO()
        self.sensor = MvCamera()
        self._config = dict()

        self._data_buffer = None
        self._hik_frame_info = MV_FRAME_OUT_INFO_EX()

    @staticmethod
    def list_device() -> list:
        device_infos = []
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            return device_infos
        if deviceList.nDeviceNum == 0:
            print("find no device!")
            return device_infos
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(
                MV_CC_DEVICE_INFO)).contents
            device_infos.append(mvcc_dev_info)
        return device_infos

    @staticmethod
    def find_device(sn) -> Tuple[bool, MV_CC_DEVICE_INFO]:
        ret_flag, device_info = False, MV_CC_DEVICE_INFO()
        device_infos = Sensor2D_Hik.list_device()
        for dev_info in device_infos:
            hik_sn = ''
            if dev_info.nTLayerType == MV_GIGE_DEVICE:
                hik_sn = dev_info.SpecialInfo.stGigEInfo.chSerialNumber
            elif dev_info.nTLayerType == MV_USB_DEVICE:
                hik_sn = dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber
            strSerialNumber = ""
            for per in hik_sn:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            if strSerialNumber == sn:
                ret_flag = True
                device_info = dev_info
                break
        return ret_flag, device_info

    @staticmethod
    def get_device_infos(dev):
        dev_info = dev
        hik_sn = ''
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            hik_sn = dev_info.SpecialInfo.stGigEInfo.chSerialNumber
        elif dev_info.nTLayerType == MV_USB_DEVICE:
            hik_sn = dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber
        strSerialNumber = ""
        for per in hik_sn:
            if per == 0:
                break
            strSerialNumber = strSerialNumber + chr(per)
        my_dev_infos = DeviceInfos()
        my_dev_infos.sn = my_dev_infos.sensor_name = strSerialNumber
        return True, my_dev_infos

    def connect(self, hik_device_info, **kwargs):
        ret = self.sensor.MV_CC_CreateHandle(hik_device_info)
        if ret != 0:
            return False
        self._device_info = hik_device_info
        ret, my_dev_info = Sensor2D_Hik.get_device_infos(hik_device_info)
        if "sensor_config" in kwargs:
            config_param = kwargs.get(
                "sensor_config", {}).get(f"{my_dev_info.sn}", {})
        else:
            config_param = dict()
        self.param.sn = my_dev_info.sn
        self.param.sensor_name = config_param.get("sensor_name", "hik_2d")
        return True

    def disconnect(self):
        ret = self.sensor.MV_CC_DestroyHandle()
        if ret != 0:
            return False
        return True

    def open(self, **kwargs):
        ret = self.sensor.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            return False
        ret = self.sync_update_config()
        if not ret:
            return False
        ret = self.set_default_config()

        # init frame parameter
        if "sensor_config" in kwargs:
            config = kwargs["sensor_config"].get(self.param.sn, {})
            frame = self.status.frame
            frame.intrinsic_matrix = np.array(config.get(
                "intrinsic_matrix", frame.intrinsic_matrix))
            frame.distortion = np.array(
                config.get("distortion", frame.distortion))
            frame.pose = np.array(config.get("frame_pose", frame.pose))

        # ch:开始取流 | en:Start grab image
        ret = self.sensor.MV_CC_StartGrabbing()
        if ret != 0:
            return False
        return True

    def close(self):
        if self._data_buffer is not None:
            del self._data_buffer
        ret = self.sensor.MV_CC_StopGrabbing()
        if ret != 0:
            return False
        ret = self.sensor.MV_CC_CloseDevice()
        if ret != 0:
            return False
        return True

    def capture(self, **kwargs) -> Tuple[bool, Frame]:
        pData = byref(self._data_buffer)
        ret = self.sensor.MV_CC_GetOneFrameTimeout(pData=pData, nDataSize=self._config[HikConfigName.PayloadSize],
                                                   stFrameInfo=self._hik_frame_info, nMsec=self._config[HikConfigName.Timeout_ms])

        ret_flag = ret == 0
        if not ret_flag:
            return False, Frame()
        num_channel = self._hik_frame_info.nFrameLen // (
            self._hik_frame_info.nHeight * self._hik_frame_info.nWidth)
        image_shape = (self._hik_frame_info.nHeight,
                       self._hik_frame_info.nWidth)
        if num_channel == 3:
            image_shape = image_shape + (3, )
        self.status.frame.image = np.asarray(
            self._data_buffer).reshape(image_shape)

        return ret_flag, self.status.frame

    def set_config(self, **kwargs):
        if HikConfigName.ExposureTime_us in kwargs:
            ret = self.sensor.MV_CC_SetFloatValue(
                HikConfigName.ExposureTime_us, kwargs[HikConfigName.ExposureTime_us])
            if ret != 0:
                return False

    def get_config(self, key):
        ret_flag, value = False, -1
        if key == HikConfigName.ExposureTime_us:
            stParam = MVCC_FLOATVALUE()
            ret = self.sensor.MV_CC_GetFloatValue(
                HikConfigName.ExposureTime_us, stParam)
            if ret == 0:
                ret_flag = True
                value = stParam.fCurValue
        return ret_flag, value

    def set_default_config(self):
        self._config[HikConfigName.Timeout_ms] = 1000

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if self._device_info.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.sensor.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.sensor.MV_CC_SetIntValue(
                    HikConfigName.GevSCPSPacketSize, nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                    return False
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                return False

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.sensor.MV_CC_SetEnumValue(
            HikConfigName.TriggerMode, MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            return False

    def sync_update_config(self):
        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.sensor.MV_CC_GetIntValue(HikConfigName.PayloadSize, stParam)
        if ret != 0:
            return False
        self._config[HikConfigName.PayloadSize] = stParam.nCurValue

        memset(byref(self._hik_frame_info), 0, sizeof(self._hik_frame_info))
        nDataSize = self._config[HikConfigName.PayloadSize]
        self._data_buffer = (c_ubyte * nDataSize)()
        return True

    def __del__(self):
        self.disconnect()

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
        frame = self.status.frame
        config["frame_pose"] = frame.pose.tolist()
        config["intrinsic_matrix"] = frame.intrinsic_matrix.tolist()
        config["distortion"] = frame.distortion.tolist()
        config["data_dir"] = f"{file_camera_dir}/{sn}/"

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
        frame = self.status.frame
        config["frame_pose"] = frame.pose.tolist()
        config["intrinsic_matrix"] = frame.intrinsic_matrix.tolist()
        config["distortion"] = frame.distortion.tolist()

        with open(sensor_config_path, 'w') as f:
            json.dump(config_sensor, f)


if __name__ == "__main__":
    import cv2
    from IPython import embed

    sn = "DA0235866"
    ret, dev_info = Sensor2D_Hik.find_device(sn=sn)
    if not ret:
        print(f"camera sn = {sn} not found")
        exit(0)
    sensor = Sensor2D_Hik()
    ret = sensor.connect(hik_device_info=dev_info)
    if not ret:
        print(f"connect to camera sn = {sn} failed")
        exit(0)
    ret = sensor.open()
    if not ret:
        print(f"open camera sn = {sn} failed")
        exit(0)

    sensor.set_config(ExposureTime=55000)

    ret, exposure_time_us = sensor.get_config(
        key=HikConfigName.ExposureTime_us)
    print(f"current exposure time (us): {exposure_time_us}")

    while True:
        ret, frame = sensor.capture()
        if not ret:
            print(f"camera sn = {sn} capture failed")
            break
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", frame.image)
        key = chr(cv2.waitKey(30) & 0xff)

        if key == 's':
            frame.save(f"./data/tmp/{sn}", id=-1)
        if key == 'q':
            break
        if key == 'e':
            embed()
    sensor.close()
    sensor.disconnect()
