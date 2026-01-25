# core/action_handler.py

import time
import math
import json
import ctypes
import random
import threading
from typing import Callable, Tuple, Optional

# [수정] 시리얼 통신 라이브러리 추가 (하드웨어 모드용)
import serial 

from utils.logger import logger
from utils.physics_utils import PhysicsUtils, MovementTracker

# --- Low Level Input Definition (Windows API) ---
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("u", KeyBdInput),
                ("xi", KeyBdInput)]

class InputWrapper(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", InputWrapper)]

class ActionHandler:
    """
    Control Layer의 핵심 실행부입니다.
    Intelligence Layer의 의도(Intent)를 실제 하드웨어/소프트웨어 입력으로 변환합니다.
    """

    # 키 매핑 (Scancode)
    KEY_MAP = {
        'left': 0xCB, 'right': 0xCD, 'up': 0xC8, 'down': 0xD0,
        'jump': 0x38,     # Alt
        'attack': 0x1D,   # Ctrl
        'loot': 0x2C,     # Z
        'interact': 0x39, # Space
        'rope': 0x2E,     # C (예시)
    }

    # [수정 1] __init__ 시그니처 변경 (main.py 호환)
    def __init__(self, mode: str = "SOFTWARE", serial_port: str = None, physics_config_path: str = "physics_engine.json"):
        self.mode = mode.upper()
        self.serial = None
        self._stop_event = threading.Event()
        
        # 물리 엔진 설정 로드
        self.physics_data = self._load_physics_config(physics_config_path)
        self.tracker = MovementTracker()
        
        # 물리 상수 캐싱 (기본값 안전하게 설정)
        self.walk_acc = self.physics_data.get('walk_acceleration', 15.0)
        self.max_walk_speed = self.physics_data.get('max_walk_velocity', 125.0)
        
        # 하드웨어 모드 초기화
        if self.mode == "HARDWARE":
            if serial_port:
                try:
                    self.serial = serial.Serial(serial_port, 115200, timeout=0.1)
                    logger.info(f"ActionHandler: Hardware Mode Initialized ({serial_port})")
                except Exception as e:
                    logger.error(f"Serial Connection Failed: {e}")
                    self.mode = "SOFTWARE" # 실패 시 소프트웨어 모드로 폴백
            else:
                logger.warning("Serial Port not provided. Falling back to SOFTWARE mode.")
                self.mode = "SOFTWARE"
        
        if self.mode == "SOFTWARE":
            logger.info("ActionHandler: Software Mode (DirectInput) Initialized.")

    def _load_physics_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Physics config not found. Using default values.")
            return {}

    # --- Low Level Input Methods ---

    def _send_key_software(self, key_code: int, pressed: bool):
        """DirectInput 방식 (SendInput)"""
        extra = ctypes.c_ulong(0)
        ii_ = InputWrapper()
        
        flags = 0x0008  # KEYEVENTF_SCANCODE
        if not pressed:
            flags |= 0x0002  # KEYEVENTF_KEYUP

        ii_.ki = KeyBdInput(0, key_code, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def _send_key_hardware(self, key_name: str, pressed: bool):
        """Arduino Serial 통신 방식"""
        if not self.serial: return
        
        # 프로토콜 예시: "D,jump\n" (Down), "U,jump\n" (Up)
        cmd = "D" if pressed else "U"
        payload = f"{cmd},{key_name}\n"
        try:
            self.serial.write(payload.encode())
        except Exception as e:
            logger.error(f"Serial Write Error: {e}")

    # [수정 2] 모드에 따른 분기 처리
    def key_down(self, key_name: str):
        if key_name not in self.KEY_MAP: return

        if self.mode == "SOFTWARE":
            self._send_key_software(self.KEY_MAP[key_name], True)
        elif self.mode == "HARDWARE":
            self._send_key_hardware(key_name, True)

    def key_up(self, key_name: str):
        if key_name not in self.KEY_MAP: return

        if self.mode == "SOFTWARE":
            self._send_key_software(self.KEY_MAP[key_name], False)
        elif self.mode == "HARDWARE":
            self._send_key_hardware(key_name, False)

    def press(self, key_name: str, duration: float = 0.05):
        """단발성 키 입력"""
        self.key_down(key_name)
        time.sleep(duration)
        self.key_up(key_name)
        time.sleep(0.02)

    # --- Physics & Feedback Control Logic (기존 로직 유지) ---

    def calculate_press_time(self, distance: float) -> float:
        """물리 공식 기반 필요 이동 시간 계산"""
        target_dist = abs(distance)
        time_to_max = self.max_walk_speed / self.walk_acc
        dist_to_max = 0.5 * self.walk_acc * (time_to_max ** 2)

        if target_dist <= dist_to_max:
            required_time = math.sqrt(2 * target_dist / self.walk_acc)
        else:
            remaining_dist = target_dist - dist_to_max
            time_at_max = remaining_dist / self.max_walk_speed
            required_time = time_to_max + time_at_max

        return required_time * 1.1

    def move_x(self, target_x: int, get_current_pos: Callable[[], Tuple[int, int]], tolerance: int = 10) -> bool:
        self._stop_event.clear()
        current_pos = get_current_pos()
        if not current_pos: return False

        start_x, start_y = current_pos
        distance = target_x - start_x
        
        if abs(distance) <= tolerance: return True

        direction = 'right' if distance > 0 else 'left'
        predicted_time = self.calculate_press_time(distance)
        
        logger.log_decision("ActionHandler", "MOVE", f"Move {direction}", f"Dist: {distance}px")

        self.key_down(direction)
        start_time = time.time()
        
        try:
            while time.time() - start_time < predicted_time + 1.0:
                if self._stop_event.is_set(): return False
                
                now_pos = get_current_pos()
                if not now_pos: continue

                curr_x, curr_y = now_pos
                self.tracker.update(curr_x, curr_y, time.time())
                
                dist_remain = target_x - curr_x
                if (direction == 'right' and dist_remain <= tolerance) or \
                   (direction == 'left' and dist_remain >= -tolerance):
                    return True
                
                # Stuck Check
                if self.tracker.check_stuck(time.time(), threshold_dist=5, timeout=0.5):
                    logger.warning(f"Stuck detected at {curr_x}")
                    self.key_up(direction)
                    return self._recover_from_stuck(get_current_pos)

                time.sleep(0.01)

            return False
        finally:
            self.key_up(direction)
            # 관성 제어 (소프트웨어 모드에서만 유의미할 수 있음)
            if self.mode == "SOFTWARE" and self.max_walk_speed > 100: 
                opp_key = 'left' if direction == 'right' else 'right'
                self.press(opp_key, 0.03)

    def _recover_from_stuck(self, get_current_pos: Callable) -> bool:
        self.press('jump')
        time.sleep(0.5)
        
        now_pos = get_current_pos()
        if now_pos and self.tracker.prev_pos:
            return PhysicsUtils.calc_distance(now_pos, self.tracker.prev_pos) > 10
        return False

    def jump_shot(self, direction: Optional[str] = None):
        if direction: self.key_down(direction)
        self.press('jump', 0.1)
        time.sleep(0.05)
        self.press('attack', 0.1)
        if direction:
            time.sleep(0.1)
            self.key_up(direction)

    def emergency_stop(self):
        self._stop_event.set()
        for key in self.KEY_MAP:
            self.key_up(key)
        logger.info("EMERGENCY STOP executed.")