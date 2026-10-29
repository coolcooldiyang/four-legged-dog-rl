"""HiPNUC（超核电子）HI12 系列 IMU 驱动。

## 为什么有这个文件

原来用的是 DM-IMU-L1（`imu_l1.py`，自定义 0x55AA + CRC16 协议）。
2026-10-15 换成 HiPNUC HI12 系列（型号 HI12M0-MI1-000），协议完全不同，
所以单独实现，并通过 config 里的 `IMU_MODEL` 切换，互不影响。

## 协议（实测确认）

厂家协议见 https://github.com/hipnuc/products 的 `hipnuc.h`。

帧结构（**总长固定 82 字节**，实测 20+ 个连续样本帧间距全部为 82）:

    偏移  长度  内容
    ----  ----  ------------------------------------------
    0     2     同步头  0x5A 0xA5
    2     2     长度    uint16 LE，payload 字节数 = 76
    4     2     校验    uint16（非标准 CRC16，本驱动用四元数模长代替校验）
    6     76    payload

payload（packet 0x91 = IMU 浮点数据，76 字节）:

    偏移  长度  内容
    ----  ----  ------------------------------------------
    0     1     tag = 0x91
    1     2     pps_sync_ms   uint16
    3     1     temp          int8  [℃]
    4     4     prs           float [Pa]（本型号未启用，恒 0）
    8     4     ts            uint32 [ms]
    12    12    acc[3]        float ×3  [g]
    24    12    gyr[3]        float ×3  [rad/s]
    36    12    mag[3]        float ×3  （本型号未启用，恒 0）
    48    12    rpy[3]        float ×3  [度]，顺序是 **(pitch, roll, yaw)**
    60    16    quat[4]       float ×4  **(w, x, y, z)**

## 实测验证（2026-10-15，设备静止水平）

    · 采样率           149 Hz
    · 四元数模长        1.000000（298 帧全部满足 |q|-1 < 1e-6）
    · 加速度模长        0.9959 g  ≈ 1g     ✓
    · 角速度            0.006 ~ 0.014      ✓ 静止时接近 0
    · rpy 顺序          (pitch, roll, yaw)  ✓ 与四元数反解完全吻合
    · 重力方向          euler 推 vs -acc    298 帧平均误差 0.0052  ✓

## 与 imu_l1.py 的关系

对外接口**完全一致**（`open/close/read_once/read_complete` + `.reading`，
`reading` 有 `accel/gyro/euler_deg/timestamp/complete`），
所以 `deployment.py` 不用改就能用。

⚠️ **单位假设**：`gyr` 按 **rad/s** 解释（HiPNUC 0x91 包的标准单位，
且静止时读数 0.006~0.014 符合陀螺零偏量级）。如果实机验证发现是 deg/s，
把 `GYRO_UNIT_SCALE` 改成 `math.pi / 180.0` 即可。
"""

from __future__ import annotations

import math
import os
import select
import struct
import termios
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------- 协议常量

SYNC1 = 0x5A
SYNC2 = 0xA5
SYNC = bytes((SYNC1, SYNC2))
HDR_LEN = 6
TAG_IMU_FLOAT = 0x91
PAYLOAD_LEN = 76
FRAME_LEN = HDR_LEN + PAYLOAD_LEN  # 82
#: 长度字段的合理上限（用于判断是不是假同步头）
MAX_PAYLOAD_LEN = 120

# payload 内的字段偏移（相对 payload 起点）
OFF_TAG = 0
OFF_TEMP = 3
OFF_TS = 8
OFF_ACC = 12
OFF_GYR = 24
OFF_MAG = 36
OFF_RPY = 48
OFF_QUAT = 60

#: gyr 的单位换算。
#:
#: ⚠️ **2026-10-15 修正**：HiPNUC HI12 的 0x91 包 `gyr` 字段单位是 **deg/s**，
#: 官方手册明确标注。最初这里写成 1.0（当成 rad/s），导致喂给策略的角速度
#: 大了 57.3 倍 —— 策略收到远超训练范围的"身体在疯狂旋转"信号，
#: 输出极端修正动作，**实机上电起立就会发癫**。
#:
#: 而 `deployment.py` 的约定是 rad/s（训练时 `base_ang_vel` 就是 rad/s），
#: 所以必须在这里换算。
GYRO_UNIT_SCALE = math.pi / 180.0

#: 陀螺仪一阶低通滤波系数（``y += alpha * (x - y)``）。**1.0 = 关闭（默认）**。
#:
#: ⚠️ 2026-10-15 更正：本驱动最初默认 0.5，理由是"实测陀螺噪声 0.195 rad/s，
#: 是训练噪声 0.115 的 1.7 倍"。**那个结论是错的** —— 当时把 HiPNUC 的
#: deg/s 误当成 rad/s，数值大了 57.3 倍。
#:
#: 单位修正后实测（log.csv，207 帧静止数据）：
#:
#:     实测陀螺噪声 std = 0.00464 rad/s
#:     训练加的噪声 std = 0.11547 rad/s  (均匀 ±0.2 rad/s)
#:     → 实测比训练噪声**小 25 倍**
#:
#: 既然传感器远比训练噪声干净，滤波没有收益，只会白白引入滞后。
#: 所以**默认关闭**。若将来换用噪声更大的 IMU，可把 alpha 调到 0.3~0.5。
GYRO_LPF_ALPHA = 1.0

#: 四元数模长容差。实测有效帧全部满足 |q|-1 < 1e-6，这里放宽到 0.05
#: 作为"帧是否对齐"的判据（垃圾数据会远远偏离 1.0）。
QUAT_NORM_TOL = 0.05

GRAVITY_MS2 = 9.80665


@dataclass
class HipnucReading:
    """一次 IMU 读数。字段语义和 `imu_l1.ImuReading` 保持一致。"""

    accel: Optional[Tuple[float, float, float]] = None      # [g]
    gyro: Optional[Tuple[float, float, float]] = None       # [rad/s]
    euler_deg: Optional[Tuple[float, float, float]] = None  # (roll, pitch, yaw) [度]
    quat: Optional[Tuple[float, float, float, float]] = None  # (w, x, y, z)
    mag: Optional[Tuple[float, float, float]] = None
    temperature: Optional[float] = None                     # [℃]
    timestamp: float = 0.0

    @property
    def complete(self) -> bool:
        return self.accel is not None and self.gyro is not None and self.euler_deg is not None

    def projected_gravity(self) -> Tuple[float, float, float]:
        """重力在机体系的方向（与 legged_gym 的 projected_gravity 同义）。

        优先用四元数（实测模长精确为 1，最可靠）。
        """
        if self.quat is not None:
            w, x, y, z = self.quat
            # R 的第三行取负
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            n = math.sqrt(gx * gx + gy * gy + gz * gz)
            if n > 1e-6:
                return (gx / n, gy / n, gz / n)
        if self.euler_deg is not None:
            roll, pitch, _yaw = (math.radians(v) for v in self.euler_deg)
            sr, cr = math.sin(roll), math.cos(roll)
            sp, cp = math.sin(pitch), math.cos(pitch)
            return (sp, -sr * cp, -cr * cp)
        return (0.0, 0.0, -1.0)


class HipnucImu:
    """HiPNUC HI12 系列串口驱动。

    用法与 `imu_l1.ImuSensor` 相同::

        with HipnucImu("/dev/ttyUSB0", 115200) as imu:
            reading = imu.read_complete(timeout=1.0)
            print(reading.gyro, reading.euler_deg)
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.02) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.fd: Optional[int] = None
        self._buf = bytearray()
        self.reading = HipnucReading()
        self.frames_ok = 0
        self.quat_rejects = 0      # 四元数模长超差被丢弃的帧（对齐失败/数据损坏）
        self.bytes_dropped = 0
        self.other_packets = 0   # 合法但非 0x91 的包（整帧跳过）
        self._gyro_lpf: Optional[list] = None   # 一阶低通滤波状态

    # ------------------------------------------------------------ 生命周期

    def open(self) -> "HipnucImu":
        if self.fd is not None:
            return self
        flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
        self.fd = os.open(self.port, flags)
        self._configure_port()
        termios.tcflush(self.fd, termios.TCIFLUSH)
        return self

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "HipnucImu":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------------------------------------------- 读取

    def read_once(self, max_bytes: int = 4096) -> HipnucReading:
        self.open()
        if self.fd is None:
            return self.reading
        ready, _, _ = select.select([self.fd], [], [], self.timeout)
        if ready:
            try:
                data = os.read(self.fd, max_bytes)
            except BlockingIOError:
                data = b""
            if data:
                self._buf.extend(data)
                self._parse()
        return self.reading

    def read_complete(self, timeout: float = 1.0) -> Optional[HipnucReading]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reading = self.read_once()
            if reading.complete:
                return reading
        return None

    # ---------------------------------------------------------------- 解析

    def _parse(self) -> None:
        while True:
            idx = self._buf.find(SYNC)
            if idx < 0:
                # 没有同步头：保留最后一个字节（可能是 0x5A），其余丢弃
                if len(self._buf) > 1:
                    self.bytes_dropped += len(self._buf) - 1
                    del self._buf[:-1]
                return
            if idx:
                self.bytes_dropped += idx
                del self._buf[:idx]
            if len(self._buf) < FRAME_LEN:
                return

            frame = bytes(self._buf[:FRAME_LEN])
            length = frame[2] | (frame[3] << 8)

            # 长度不合理 → 可能是假同步头，单字节重同步
            if not (1 <= length <= MAX_PAYLOAD_LEN):
                self.bytes_dropped += 1
                del self._buf[0]
                continue

            # 合法帧但不是我们要的 packet → 用长度字段整帧跳过（避免级联丢弃）
            if length != PAYLOAD_LEN or frame[HDR_LEN] != TAG_IMU_FLOAT:
                self.bytes_dropped += HDR_LEN + length
                del self._buf[:HDR_LEN + length]
                self.other_packets += 1
                continue

            payload = frame[HDR_LEN:FRAME_LEN]
            quat = struct.unpack_from("<4f", payload, OFF_QUAT)

            # 用四元数模长做帧有效性判据（实测有效帧 |q|-1 < 1e-6）
            norm = math.sqrt(sum(v * v for v in quat))
            if abs(norm - 1.0) > QUAT_NORM_TOL:
                self.quat_rejects += 1
                del self._buf[0]
                continue

            acc = struct.unpack_from("<3f", payload, OFF_ACC)
            gyr = struct.unpack_from("<3f", payload, OFF_GYR)
            mag = struct.unpack_from("<3f", payload, OFF_MAG)
            rpy = struct.unpack_from("<3f", payload, OFF_RPY)   # (pitch, roll, yaw)
            temp = struct.unpack_from("<b", payload, OFF_TEMP)[0]

            # 重排成 (roll, pitch, yaw) —— 与 imu_l1 / deployment 的约定一致
            euler_deg = (float(rpy[1]), float(rpy[0]), float(rpy[2]))

            self.reading.accel = tuple(float(v) for v in acc)
            gyro = [float(v) * GYRO_UNIT_SCALE for v in gyr]
            if GYRO_LPF_ALPHA < 1.0:
                if self._gyro_lpf is None:
                    self._gyro_lpf = list(gyro)
                else:
                    a = GYRO_LPF_ALPHA
                    self._gyro_lpf = [
                        self._gyro_lpf[i] + a * (gyro[i] - self._gyro_lpf[i])
                        for i in range(3)
                    ]
                gyro = list(self._gyro_lpf)
            self.reading.gyro = tuple(gyro)
            self.reading.euler_deg = euler_deg
            self.reading.quat = tuple(float(v) for v in quat)
            self.reading.mag = tuple(float(v) for v in mag)
            self.reading.temperature = float(temp)
            self.reading.timestamp = time.time()

            self.frames_ok += 1
            del self._buf[:FRAME_LEN]

    # ------------------------------------------------------------ 串口配置

    def _configure_port(self) -> None:
        if self.fd is None:
            return
        try:
            baud_const = getattr(termios, "B%d" % self.baudrate)
        except AttributeError:
            raise ValueError("不支持的波特率: %d" % self.baudrate)

        attrs = termios.tcgetattr(self.fd)
        # 8N1、原始模式
        attrs[0] &= ~(
            termios.BRKINT
            | termios.ICRNL
            | termios.INPCK
            | termios.ISTRIP
            | termios.IXON
        )
        for name in ("IXOFF", "IXANY"):
            if hasattr(termios, name):
                attrs[0] &= ~getattr(termios, name)
        attrs[0] |= termios.CLOCAL | termios.CREAD
        attrs[1] &= ~termios.OPOST
        attrs[2] |= termios.CS8
        attrs[2] &= ~(termios.PARENB | termios.CSTOPB)
        if hasattr(termios, "CRTSCTS"):
            attrs[2] &= ~termios.CRTSCTS
        attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ISIG)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        attrs[4] = attrs[5] = baud_const

        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIFLUSH)


def quat_to_euler_deg(quat) -> Tuple[float, float, float]:
    """四元数 (w,x,y,z) → (roll, pitch, yaw) 度。仅供自检/调试用。"""
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
