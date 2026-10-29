from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lib.pub_user import DamiaoDevice


@dataclass
class CanFrame:
    serial: str
    device_index: int
    channel: int
    can_id: int
    data: bytes
    timestamp: float
    is_fd: bool = False
    is_extended: bool = False
    is_tx_echo: bool = False


class Usb2CanHub:
    def __init__(
        self,
        serials: Optional[Sequence[str]] = None,
        channels: Sequence[int] = (0, 1),
        bitrate: int = 1000000,
        canfd: bool = False,
        rx_queue_size: int = 4096,
    ) -> None:
        self.serials = list(serials or [])
        self.channels = tuple(int(ch) for ch in channels)
        self.bitrate = int(bitrate)
        self.canfd = bool(canfd)
        self.dm = DamiaoDevice()
        self.handle = None
        self.devices: List[Dict[str, object]] = []
        self.rx_queue: "queue.Queue[CanFrame]" = queue.Queue(maxsize=rx_queue_size)

    def open(self) -> "Usb2CanHub":
        self.handle = self.dm.create_handle()
        count = self.dm.find_devices()
        if count <= 0:
            raise RuntimeError("no DaMiao USB2FDCAN devices found")

        all_devices = self.dm.get_devices()
        selected = []
        for dev_index, dev in enumerate(all_devices):
            opened = self.dm.open_device(dev)
            sn = self.dm.get_sn(dev)
            if self.serials and sn not in self.serials:
                if opened:
                    self.dm.close_device(dev)
                continue
            if not opened:
                raise RuntimeError("failed to open USB2FDCAN device %s" % sn)
            selected.append((dev_index, dev, sn))

        missing = [sn for sn in self.serials if sn not in [item[2] for item in selected]]
        if missing:
            raise RuntimeError("missing USB2FDCAN devices: %s" % ", ".join(missing))

        for dev_index, dev, sn in selected:
            channels_opened: List[int] = []
            self.dm.hook_to_rec(dev, self._make_rx_callback(dev_index, sn))
            for ch in self.channels:
                if not self.dm.set_baud(dev, ch, canfd=self.canfd, bitrate=self.bitrate, dbitrate=self.bitrate):
                    raise RuntimeError("failed to set CAN baud: %s ch%d" % (sn, ch))
                if not self.dm.open_channel(dev, ch):
                    raise RuntimeError("failed to open CAN channel: %s ch%d" % (sn, ch))
                channels_opened.append(ch)
            self.devices.append({"index": dev_index, "handle": dev, "serial": sn, "channels": channels_opened})
        return self

    def close(self) -> None:
        for info in reversed(self.devices):
            dev = info["handle"]
            for ch in info["channels"]:
                try:
                    self.dm.close_channel(dev, ch)
                except Exception:
                    pass
            try:
                self.dm.close_device(dev)
            except Exception:
                pass
        self.devices.clear()
        # Do not call damiao_handle_destroy here. The shipped SDK has shown
        # double-free crashes during repeated Python open/close cycles.
        self.handle = None

    def __enter__(self) -> "Usb2CanHub":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def send(
        self,
        serial: str,
        channel: int,
        can_id: int,
        data: Iterable[int],
        *,
        ext: bool = False,
        canfd: Optional[bool] = None,
        brs: bool = False,
        count: int = 1,
    ) -> None:
        dev = self._find_device(serial)
        payload = list(data)
        self.dm.send_fast(dev, int(channel), int(can_id), int(count), bool(ext), self.canfd if canfd is None else bool(canfd), bool(brs), payload)

    def recv(self, timeout: float = 0.0) -> Optional[CanFrame]:
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, limit: int = 256) -> List[CanFrame]:
        frames = []
        for _ in range(limit):
            frame = self.recv(timeout=0.0)
            if frame is None:
                break
            frames.append(frame)
        return frames

    def describe(self) -> List[Tuple[int, str, Tuple[int, ...]]]:
        return [(int(info["index"]), str(info["serial"]), tuple(info["channels"])) for info in self.devices]

    def _find_device(self, serial: str):
        for info in self.devices:
            if info["serial"] == serial:
                return info["handle"]
        raise KeyError("unknown CAN device serial: %s" % serial)

    def _make_rx_callback(self, dev_index: int, serial: str):
        def _callback(frame_ptr):
            frame = frame_ptr.contents
            dlc = int(frame.head.dlc)
            data = bytes(frame.payload[:dlc])
            item = CanFrame(
                serial=serial,
                device_index=dev_index,
                channel=int(frame.head.channel),
                can_id=int(frame.head.can_id),
                data=data,
                timestamp=time.time(),
                is_fd=bool(frame.head.canfd),
                is_extended=bool(frame.head.ext),
                is_tx_echo=bool(frame.head.dir),
            )
            try:
                self.rx_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.rx_queue.get_nowait()
                except queue.Empty:
                    pass
                self.rx_queue.put_nowait(item)

        return _callback
