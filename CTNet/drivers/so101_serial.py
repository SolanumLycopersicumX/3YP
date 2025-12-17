import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import serial  # pyserial
except Exception:
    serial = None


# Feetech SCServo/STS Protocol 0 (Dynamixel v1-like)
_HDR = b"\xFF\xFF"


def _lo(x: int) -> int:
    return x & 0xFF


def _hi(x: int) -> int:
    return (x >> 8) & 0xFF


def _checksum(payload: bytes) -> int:
    # checksum over bytes from ID to last param
    s = sum(payload) & 0xFF
    return (~s) & 0xFF


@dataclass
class Limits:
    min_ticks: Optional[int] = None
    max_ticks: Optional[int] = None

    def clamp(self, ticks: int) -> int:
        if self.min_ticks is not None:
            ticks = max(ticks, int(self.min_ticks))
        if self.max_ticks is not None:
            ticks = min(ticks, int(self.max_ticks))
        return ticks


class So101Bus:
    """Minimal Feetech STS3215 bus driver (Protocol 0).

    - Default baud: 1_000_000 (1 Mbps)
    - Timeout: 20 ms typical
    - Little-endian for 2-byte values (Lo, Hi).
    """

    # Control table (decimal addresses)
    TORQUE_ENABLE = 40
    GOAL_POSITION = 42  # 2 bytes
    GOAL_TIME = 44      # 2 bytes (ms)
    GOAL_VELOCITY = 46  # 2 bytes
    PRESENT_POSITION = 56  # 2 bytes
    LOCK = 55
    OPERATING_MODE = 33
    RETURN_DELAY_TIME = 7
    MIN_POSITION_LIMIT = 9
    MAX_POSITION_LIMIT = 11

    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03

    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.02, debug: bool = False):
        if serial is None:
            raise ImportError("pyserial 未安装，请先: pip install pyserial")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.debug = debug
        self._ser: Optional[serial.Serial] = None

    # ---- lifecycle ----
    def open(self):
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout, write_timeout=self.timeout, bytesize=8, parity='N', stopbits=1)

    def close(self):
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    # ---- low-level ----
    def _write_frame(self, dev_id: int, inst: int, params: bytes) -> None:
        length = 2 + len(params)  # INST + PARAMS + CHKSUM
        payload = bytes([dev_id & 0xFF, length & 0xFF, inst & 0xFF]) + params
        cs = _checksum(payload)
        frame = _HDR + payload + bytes([cs])
        if self.debug:
            print("TX:", frame.hex(' '))
        assert self._ser is not None
        self._ser.reset_input_buffer()
        self._ser.write(frame)

    def _read_status(self, expected_id: Optional[int] = None) -> Tuple[int, int, bytes]:
        assert self._ser is not None
        # seek header
        t0 = time.time()
        b0 = self._ser.read(1)
        while b0 != b"\xFF":
            if (time.time() - t0) > self.timeout:
                raise TimeoutError("等待状态帧超时(头1)")
            b0 = self._ser.read(1)
        b1 = self._ser.read(1)
        if b1 != b"\xFF":
            # restart search
            return self._read_status(expected_id)
        # id, length, error
        hdr = self._ser.read(3)
        if len(hdr) < 3:
            raise TimeoutError("等待状态帧超时(头2)")
        dev_id = hdr[0]
        length = hdr[1]
        error = hdr[2]
        # params + checksum
        n = max(0, length - 2)  # exclude error + checksum
        params = self._ser.read(n)
        cs = self._ser.read(1)
        if len(params) < n or len(cs) < 1:
            raise TimeoutError("等待状态帧超时(数据)")
        payload = bytes([dev_id, length, error]) + params
        if _checksum(payload) != cs[0]:
            raise ValueError("状态帧校验失败")
        if expected_id is not None and dev_id != (expected_id & 0xFF):
            raise ValueError(f"状态帧ID不符，期望{expected_id}，收到{dev_id}")
        if self.debug:
            raw = _HDR + payload + cs
            print("RX:", raw.hex(' '))
        return dev_id, error, params

    # ---- basic instructions ----
    def ping(self, dev_id: int) -> bool:
        self._write_frame(dev_id, self.INST_PING, b"")
        try:
            _, err, _ = self._read_status(expected_id=dev_id)
            return err == 0
        except Exception:
            return False

    def read(self, dev_id: int, addr: int, length: int) -> bytes:
        params = bytes([addr & 0xFF, length & 0xFF])
        self._write_frame(dev_id, self.INST_READ, params)
        _id, err, data = self._read_status(expected_id=dev_id)
        if err != 0:
            raise RuntimeError(f"设备{dev_id} 读错误: error=0x{err:02X}")
        return data

    def write(self, dev_id: int, addr: int, data: bytes) -> None:
        params = bytes([addr & 0xFF]) + data
        self._write_frame(dev_id, self.INST_WRITE, params)
        _id, err, _ = self._read_status(expected_id=dev_id)
        if err != 0:
            raise RuntimeError(f"设备{dev_id} 写错误: error=0x{err:02X}")

    # ---- helpers ----
    @staticmethod
    def ticks_to_deg(ticks: int) -> float:
        return float(ticks) * 360.0 / 4096.0

    @staticmethod
    def deg_to_ticks(deg: float) -> int:
        return int(round(float(deg) * 4096.0 / 360.0))

    @staticmethod
    def rad_to_ticks(rad: float) -> int:
        return int(round(float(rad) * 4096.0 / (2.0 * math.pi)))

    # ---- high-level ----
    def torque_enable(self, dev_id: int, on: bool) -> None:
        self.write(dev_id, self.TORQUE_ENABLE, bytes([1 if on else 0]))

    def set_operating_mode(self, dev_id: int, mode: int = 0) -> None:
        self.write(dev_id, self.OPERATING_MODE, bytes([mode & 0xFF]))

    def set_return_delay(self, dev_id: int, val: int = 0) -> None:
        self.write(dev_id, self.RETURN_DELAY_TIME, bytes([val & 0xFF]))

    def read_position(self, dev_id: int) -> int:
        data = self.read(dev_id, self.PRESENT_POSITION, 2)
        return int(data[0]) | (int(data[1]) << 8)

    def write_position(self, dev_id: int, pos_ticks: int, time_ms: Optional[int] = None, velocity: Optional[int] = None) -> None:
        """Set target position; if time/velocity provided, write them first so they apply to this move."""
        if time_ms is not None:
            t = max(0, min(0xFFFF, int(time_ms)))
            self.write(dev_id, self.GOAL_TIME, bytes([_lo(t), _hi(t)]))
        if velocity is not None:
            v = max(0, min(0xFFFF, int(velocity)))
            self.write(dev_id, self.GOAL_VELOCITY, bytes([_lo(v), _hi(v)]))
        self.write(dev_id, self.GOAL_POSITION, bytes([_lo(pos_ticks), _hi(pos_ticks)]))


@dataclass
class So101Map:
    """关节名到ID与限位的映射。"""
    name_to_id: Dict[str, int]
    limits: Dict[str, Limits]

    @staticmethod
    def default() -> "So101Map":
        # 1..6 默认映射
        names = [
            ("shoulder_pan", 1),
            ("shoulder_lift", 2),
            ("elbow_flex", 3),
            ("wrist_flex", 4),
            ("wrist_roll", 5),
            ("gripper", 6),
        ]
        return So101Map(dict(names), {n: Limits() for n, _ in names})
