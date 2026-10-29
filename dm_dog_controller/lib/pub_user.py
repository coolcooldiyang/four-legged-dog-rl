import ctypes
import os
import sys
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64, c_int, c_int32, c_float, c_bool, c_void_p, c_char_p, c_size_t, Structure, POINTER, CFUNCTYPE
# ================= 配置区域 =================
# 请修改这里指向你的 .so 文件路径，如果是绝对路径
def _resolve_lib_path():
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdm_device.so")
    if os.path.exists(local_path):
        return local_path
    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = os.path.join(get_package_share_directory("dm_dog_controller"), "lib", "libdm_device.so")
        if os.path.exists(share_path):
            return share_path
    except Exception:
        pass
    return local_path


LIB_PATH = _resolve_lib_path()
# 如果库在系统路径中，直接写名字即可，例如 "dm_device"
# ============================================

# 1. 定义 C 语言对应的数据结构
class UsbTxFrameHead(Structure):
    _pack_ = 1  # 对应 #pragma pack(push,1)
    _fields_ = [
        ("can_id", c_uint32, 29),
        ("esi", c_uint32, 1),
        ("ext", c_uint32, 1),
        ("rtr", c_uint32, 1),
        ("canfd", c_uint8, 1),
        ("brs", c_uint8, 1),
        ("id_inc", c_uint8, 1),
        ("data_inc", c_uint8, 1),
        ("dlc", c_uint8, 4),
        ("channel", c_uint8),
        ("reserved", c_uint16),
        ("step_id", c_uint16),
        ("stop_id", c_uint32),
        ("interval", c_uint32),
        ("send_times", c_int),
    ]

class UsbTxFrame(Structure):
    _pack_ = 1
    _fields_ = [
        ("head", UsbTxFrameHead),
        ("payload", c_uint8 * 64),
    ]

class UsbRxFrameHead(Structure):
    _pack_ = 1
    _fields_ = [
        ("can_id", c_uint32, 29),
        ("esi", c_uint32, 1),
        ("ext", c_uint32, 1),
        ("rtr", c_uint32, 1),
        ("time_stamp", c_uint64),
        ("channel", c_uint8),
        ("canfd", c_uint8, 1),
        ("dir", c_uint8, 1),
        ("brs", c_uint8, 1),
        ("ack", c_uint8, 1),
        ("dlc", c_uint8, 4),
        ("reserved", c_uint16),
    ]

class UsbRxFrame(Structure):
    _pack_ = 1
    _fields_ = [
        ("head", UsbRxFrameHead),
        ("payload", c_uint8 * 64),
    ]

class DeviceBaud(Structure):
    _fields_ = [
        ("can_baudrate", c_int),
        ("canfd_baudrate", c_int),
        ("can_sp", c_float),
        ("canfd_sp", c_float),
    ]

# 2. 定义枚举和回调类型
DEV_USB2CANFD_DUAL = 1  # 对应头文件中的 enum
DAMIAO_USB_VENDOR_ID = "34b7"

# 定义回调函数类型
REC_CALLBACK_TYPE = CFUNCTYPE(None, POINTER(UsbRxFrame))
SENT_CALLBACK_TYPE = CFUNCTYPE(None, POINTER(UsbRxFrame))
ERR_CALLBACK_TYPE = CFUNCTYPE(None, POINTER(UsbRxFrame))


def _read_sysfs_value(path):
    try:
        with open(path, "r", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def _find_damiao_usb_nodes():
    devices = []
    usb_root = "/sys/bus/usb/devices"
    try:
        names = os.listdir(usb_root)
    except OSError:
        return devices

    for name in names:
        sys_path = os.path.join(usb_root, name)
        vendor_id = _read_sysfs_value(os.path.join(sys_path, "idVendor"))
        if vendor_id != DAMIAO_USB_VENDOR_ID:
            continue

        busnum = _read_sysfs_value(os.path.join(sys_path, "busnum"))
        devnum = _read_sysfs_value(os.path.join(sys_path, "devnum"))
        product_id = _read_sysfs_value(os.path.join(sys_path, "idProduct")) or "unknown"
        if not busnum or not devnum:
            continue

        node = f"/dev/bus/usb/{int(busnum):03d}/{int(devnum):03d}"
        devices.append((node, product_id))
    return devices


def _ensure_damiao_usb_permissions():
    blocked = [
        (node, product_id)
        for node, product_id in _find_damiao_usb_nodes()
        if not os.access(node, os.R_OK | os.W_OK)
    ]
    if not blocked:
        return

    node_list = ", ".join(f"{node} ({DAMIAO_USB_VENDOR_ID}:{product_id})" for node, product_id in blocked)
    raise PermissionError(
        "DaMiao USB device is not readable/writable by the current user: "
        f"{node_list}\n"
        "Fix once with:\n"
        "  echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"34b7\", ATTR{idProduct}==\"6632\", GROUP=\"plugdev\", MODE=\"0660\", TAG+=\"uaccess\"' | sudo tee /etc/udev/rules.d/99-damiao.rules\n"
        "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
        "Then unplug and plug the device again."
    )


class DamiaoDevice:
    def __init__(self):
        # --- 新增代码开始 ---
        # 1. 先加载 libusb 依赖库，使用 RTLD_GLOBAL 使得后续加载的库可以使用它
        try:
            # 尝试加载系统 libusb-1.0
            # 注意：在 Linux 下通常叫 libusb-1.0.so.0 或 libusb-1.0.so
            ctypes.CDLL('libusb-1.0.so.0', ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                ctypes.CDLL('libusb-1.0.so', ctypes.RTLD_GLOBAL)
            except OSError:
                print("Warning: Could not find libusb, SDK might fail.")
        # --- 新增代码结束 ---

        # 2. 加载你的 SDK 库
        if not os.path.exists(LIB_PATH):
            print(f"Error: Library not found at {LIB_PATH}")
            try:
                self.lib = ctypes.CDLL("libdm_device.so")
            except:
                raise Exception("Failed to load library. Please check the path.")
        else:
            self.lib = ctypes.CDLL(LIB_PATH)

        self.handle = None
        self.dev_list = None
        self.dev_count = 0
        self._can_destroy_handle = False
        
        # 保持回调函数引用，防止被垃圾回收
        self._rec_callback_refs = []
        self._sent_callback_refs = []

        self._setup_api()

    def _setup_api(self):
        """配置 C 函数的参数和返回类型"""
        
        # ==========================================
        # 重点修复区域：必须指定返回值为 c_void_p (指针)
        # ==========================================
        
        # 1. damiao_handle_create
        # 错误原因：默认返回 int，导致 64位地址被截断
        self.lib.damiao_handle_create.argtypes = [c_int]
        self.lib.damiao_handle_create.restype = c_void_p  # <--- 必须加这个！

        self.lib.damiao_handle_get_devices.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_int)]
        self.lib.damiao_handle_get_devices.restype = None

        



        # ==========================================
        # 其他函数配置
        # ==========================================

        # damiao_handle_destroy
        self.lib.damiao_handle_destroy.argtypes = [c_void_p]
        self.lib.damiao_handle_destroy.restype = None
        
        # damiao_print_version
        self.lib.damiao_print_version.argtypes = [c_void_p]
        self.lib.damiao_print_version.restype = None

        # damiao_handle_find_devices
        self.lib.damiao_handle_find_devices.argtypes = [c_void_p]
        self.lib.damiao_handle_find_devices.restype = c_int

        # device_open
        self.lib.device_open.argtypes = [c_void_p]
        self.lib.device_open.restype = c_bool

        # device_close
        self.lib.device_close.argtypes = [c_void_p]
        self.lib.device_close.restype = c_bool

        # device_get_version
        self.lib.device_get_version.argtypes = [c_void_p, c_char_p, c_size_t]
        self.lib.device_get_version.restype = None

        # device_get_serial_number
        self.lib.device_get_serial_number.argtypes = [c_void_p, c_char_p, c_size_t]
        self.lib.device_get_serial_number.restype = None

        # device_open_channel
        self.lib.device_open_channel.argtypes = [c_void_p, c_uint8]
        self.lib.device_open_channel.restype = c_bool

        # device_close_channel
        self.lib.device_close_channel.argtypes = [c_void_p, c_uint8]
        self.lib.device_close_channel.restype = c_bool

        # device_channel_set_baud_with_sp
        self.lib.device_channel_set_baud_with_sp.argtypes = [c_void_p, c_uint8, c_bool, c_int, c_int, c_float, c_float]
        self.lib.device_channel_set_baud_with_sp.restype = c_bool

        # device_channel_get_baudrate
        self.lib.device_channel_get_baudrate.argtypes = [c_void_p, c_uint8, POINTER(DeviceBaud)]
        self.lib.device_channel_get_baudrate.restype = c_bool

        # device_hook_to_rec
        self.lib.device_hook_to_rec.argtypes = [c_void_p, REC_CALLBACK_TYPE]
        self.lib.device_hook_to_rec.restype = None

        # device_hook_to_sent
        self.lib.device_hook_to_sent.argtypes = [c_void_p, SENT_CALLBACK_TYPE]
        self.lib.device_hook_to_sent.restype = None

        # device_channel_send_fast
        self.lib.device_channel_send_fast.argtypes = [c_void_p, c_uint8, c_uint32, c_int32, c_bool, c_bool, c_bool, c_uint8, POINTER(c_uint8)]
        self.lib.device_channel_send_fast.restype = None

    # ================= 封装的高级接口 =================

    def create_handle(self, dev_type=DEV_USB2CANFD_DUAL):
        self.handle = self.lib.damiao_handle_create(dev_type)
        if not self.handle:
            raise Exception("Failed to create handle")
        return self.handle

    def print_version(self):
        if self.handle:
            self.lib.damiao_print_version(self.handle)

    def find_devices(self):
        if not self.handle: return 0
        try:
            _ensure_damiao_usb_permissions()
        except PermissionError:
            self._can_destroy_handle = False
            raise
        self.dev_count = self.lib.damiao_handle_find_devices(self.handle)
        self._can_destroy_handle = True
        return self.dev_count

    def get_devices(self):
        if not self.handle: return []
        
        # 创建一个指针数组来接收设备句柄
        # 假设最多16个设备，对应 C 代码中的 dev_list[16]
        device_array = (c_void_p * 16)()
        count = c_int(0)
        
        self.lib.damiao_handle_get_devices(self.handle, device_array, ctypes.byref(count))
        
        devices = []
        for i in range(count.value):
            devices.append(device_array[i])
        return devices

    def open_device(self, dev_handle):
        return self.lib.device_open(dev_handle)

    def close_device(self, dev_handle):
        return self.lib.device_close(dev_handle)

    def get_version(self, dev_handle):
        buf = ctypes.create_string_buffer(255)
        self.lib.device_get_version(dev_handle, buf, 255)
        return buf.value.decode('utf-8')

    def get_sn(self, dev_handle):
        buf = ctypes.create_string_buffer(255)
        self.lib.device_get_serial_number(dev_handle, buf, 255)
        return buf.value.decode('utf-8')

    def set_baud(self, dev_handle, channel, canfd=True, bitrate=1000000, dbitrate=5000000, sp=0.75):
        return self.lib.device_channel_set_baud_with_sp(dev_handle, channel, canfd, bitrate, dbitrate, sp, sp)

    def get_baud(self, dev_handle, channel):
        baud = DeviceBaud()
        if self.lib.device_channel_get_baudrate(dev_handle, channel, ctypes.byref(baud)):
            return {
                "can_baud": baud.can_baudrate,
                "canfd_baud": baud.canfd_baudrate,
                "can_sp": baud.can_sp,
                "canfd_sp": baud.canfd_sp
            }
        return None

    def open_channel(self, dev_handle, channel):
        return self.lib.device_open_channel(dev_handle, channel)

    def close_channel(self, dev_handle, channel):
        return self.lib.device_close_channel(dev_handle, channel)

    def send_fast(self, dev_handle, ch, can_id, cnt, ext, canfd, brs, payload_list):
        """
        payload_list: Python list of integers (e.g., [1, 2, 3, 4, 5, 6, 7, 8])
        """
        if len(payload_list) > 64:
            raise ValueError("CAN FD payload must be <= 64 bytes")
        if not canfd and len(payload_list) > 8:
            raise ValueError("CAN 2.0 payload must be <= 8 bytes")
        for value in payload_list:
            if value < 0 or value > 255:
                raise ValueError("payload values must be in range 0..255")

        payload_array = (c_uint8 * len(payload_list))(*payload_list)
        self.lib.device_channel_send_fast(dev_handle, ch, can_id, cnt, ext, canfd, brs, len(payload_list), payload_array)

    def hook_to_rec(self, dev_handle, callback_func):
        """
        callback_func: Python function accepting a UsbRxFrame pointer
        """
        callback_ref = REC_CALLBACK_TYPE(callback_func)
        self._rec_callback_refs.append(callback_ref)
        if dev_handle:
            self.lib.device_hook_to_rec(dev_handle, callback_ref)

    def hook_to_sent(self, dev_handle, callback_func):
        callback_ref = SENT_CALLBACK_TYPE(callback_func)
        self._sent_callback_refs.append(callback_ref)
        if dev_handle:
            self.lib.device_hook_to_sent(dev_handle, callback_ref)

    def destroy(self):
        if self.handle:
            if not self._can_destroy_handle:
                # The SDK may crash when destroying a handle before a successful discovery pass.
                self.handle = None
                return
            self.lib.damiao_handle_destroy(self.handle)
            self.handle = None

# ================= 测试主程序 =================
if __name__ == "__main__":
    print("--- Python Damiao SDK Test ---")
    CAN_CHANNELS = (0, 1)  # SDK channel 0/1 = physical channel 1/2
    CLASSIC_CAN_BITRATE = 1000000
    CAN_ID = 0x123
    
    # 1. 初始化
    dm = DamiaoDevice()
    opened_channels = []
    dev = None
    try:
        dm.create_handle()
        dm.print_version()
        
        # 2. 查找设备
        cnt = dm.find_devices()
        if cnt == 0:
            print("No device found!")
            sys.exit(0)
            
        # 3. 获取设备列表
        devices = dm.get_devices()
        dev = devices[0] # 取第一个设备
        
        # 4. 打开设备
        if dm.open_device(dev):
            print("Device opened!")
            
            # 5. 获取信息
            ver = dm.get_version(dev)
            sn = dm.get_sn(dev)
            print(f"Version: {ver}")
            print(f"SN: {sn}")
            
            # 6. 配置经典 CAN 1000k 波特率，物理通道 1/2 对应 SDK 通道 0/1
            for channel in CAN_CHANNELS:
                if dm.set_baud(dev, channel, canfd=False, bitrate=CLASSIC_CAN_BITRATE, dbitrate=CLASSIC_CAN_BITRATE):
                    print(f"Channel {channel + 1} classic CAN bitrate configured: {CLASSIC_CAN_BITRATE}")
            if os.environ.get("DM_QUERY_BAUD") == "1":
                for channel in CAN_CHANNELS:
                    baud_info = dm.get_baud(dev, channel)
                    if baud_info:
                        print(f"Channel {channel + 1} Baudrate Info: {baud_info}")
                
            # 7. 打开通道 1/2
            for channel in CAN_CHANNELS:
                if dm.open_channel(dev, channel):
                    opened_channels.append(channel)
                    print(f"Channel {channel + 1} opened.")
                else:
                    print(f"Open channel {channel + 1} failed!")
            
            # 8. 注册回调 (示例)
            def my_rec_callback(frame_ptr):
                frame = frame_ptr.contents
                print(f"Received CH{frame.head.channel + 1} CAN ID: {hex(frame.head.can_id)}, DLC: {frame.head.dlc}")
                # 注意：在回调中访问 payload 需要小心，这里仅打印长度
            
            dm.hook_to_rec(dev, my_rec_callback)
            
            # 9. 两个通道都发送经典 CAN 数据帧
            payload = [1, 2, 3, 4, 5, 6, 7, 8]
            for channel in opened_channels:
                print(f"Sending classic CAN data on channel {channel + 1}...")
                dm.send_fast(dev, channel, CAN_ID, 1, False, False, False, payload)
            
            # 保持运行一会儿以接收回环数据（如果有）
            import time
            time.sleep(1)
            
            # 10. 清理
            for channel in opened_channels:
                dm.close_channel(dev, channel)
            dm.close_device(dev)
            print("Device closed.")
        else:
            print("Open device failed!")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        dm.destroy()
