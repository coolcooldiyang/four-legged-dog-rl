from __future__ import annotations

import math
import os
import select
import struct
import termios
import time
from dataclasses import dataclass
from typing import Optional, Tuple

HDR = b"\x55\xAA"
TAIL = 0x0A
FRAME_LEN = 19
REG_ACCEL = 0x01
REG_GYRO = 0x02
REG_EULER = 0x03


@dataclass
class ImuReading:
    accel: Optional[Tuple[float, float, float]] = None
    gyro: Optional[Tuple[float, float, float]] = None
    euler_deg: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0

    @property
    def complete(self) -> bool:
        return self.accel is not None and self.gyro is not None and self.euler_deg is not None


class ImuSensor:
    def __init__(self, port: str, baudrate: int = 921600, timeout: float = 0.02) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.fd: Optional[int] = None
        self._buf = bytearray()
        self.reading = ImuReading()
        self.frames_ok = 0
        self.crc_errors = 0
        self.bytes_dropped = 0
        self._crc16 = _load_dm_crc16()

    def open(self) -> "ImuSensor":
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

    def __enter__(self) -> "ImuSensor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_once(self, max_bytes: int = 4096) -> ImuReading:
        self.open()
        if self.fd is None:
            return self.reading

        ready, _, _ = select.select([self.fd], [], [], self.timeout)
        if ready:
            data = os.read(self.fd, max_bytes)
            if data:
                self._buf.extend(data)
                self._parse()
        return self.reading

    def read_complete(self, timeout: float = 1.0) -> Optional[ImuReading]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reading = self.read_once()
            if reading.complete:
                return reading
        return None

    def _parse(self) -> None:
        while True:
            idx = self._buf.find(HDR)
            if idx < 0:
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
            if frame[-1] != TAIL:
                self.bytes_dropped += 1
                del self._buf[0]
                continue

            reg = frame[3]
            if reg not in (REG_ACCEL, REG_GYRO, REG_EULER):
                self.bytes_dropped += 1
                del self._buf[0]
                continue

            crc_wire = frame[16] | (frame[17] << 8)
            crc_ok = self._crc16(frame[:16]) == crc_wire or self._crc16(frame[2:16]) == crc_wire
            if not crc_ok:
                self.crc_errors += 1
                del self._buf[0]
                continue

            values = struct.unpack("<fff", frame[4:16])
            now = time.time()
            if reg == REG_ACCEL:
                self.reading.accel = values
            elif reg == REG_GYRO:
                self.reading.gyro = values
            elif reg == REG_EULER:
                self.reading.euler_deg = values
            self.reading.timestamp = now
            self.frames_ok += 1
            del self._buf[:FRAME_LEN]

    def _configure_port(self) -> None:
        if self.fd is None:
            return
        baud_const = getattr(termios, "B%d" % self.baudrate)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        if hasattr(termios, "IXOFF"):
            attrs[0] &= ~termios.IXOFF
        if hasattr(termios, "IXANY"):
            attrs[0] &= ~termios.IXANY
        attrs[1] = 0
        attrs[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
        attrs[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
        if hasattr(termios, "CRTSCTS"):
            attrs[2] &= ~termios.CRTSCTS
        attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ISIG)
        if hasattr(termios, "IEXTEN"):
            attrs[3] &= ~termios.IEXTEN
        attrs[4] = baud_const
        attrs[5] = baud_const
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)


def euler_deg_to_projected_gravity(euler_deg: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if euler_deg is None:
        return (0.0, 0.0, -1.0)
    roll, pitch, _yaw = [math.radians(v) for v in euler_deg]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    # R_body^world.T @ [0, 0, -1], yaw cancels for gravity.
    return (sp, -sr * cp, -cr * cp)


def _load_dm_crc16():
    try:
        from bs_hardware.dm_crc import dm_crc16
    except Exception as exc:
        raise RuntimeError("DM IMU CRC implementation is not available in bs_hardware.dm_crc") from exc
    return dm_crc16
