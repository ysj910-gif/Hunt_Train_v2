# core/action_handler.py

import time
import math
import json
import ctypes
import threading
from typing import Callable, Tuple, Optional
# import serial 
from utils.logger import logger
from utils.physics_utils import PhysicsUtils, MovementTracker
from utils.logger import logger, trace_logic
from utils.human_input import HumanInput
from core.arduino_bridge import ArduinoBridge
from core.latency_monitor import latency_monitor 


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
    
    # [수정] 확장된 DirectInput 스캔코드 매핑 (대소문자 무관 처리를 위해 소문자로 작성)
    SCANCODE_MAP = {
        'esc': 0x01, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06, '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A, '0': 0x0B,
        '-': 0x0C, '=': 0x0D, 'backspace': 0x0E, 'tab': 0x0F,
        'q': 0x10, 'w': 0x11, 'e': 0x12, 'r': 0x13, 't': 0x14, 'y': 0x15, 'u': 0x16, 'i': 0x17, 'o': 0x18, 'p': 0x19, '[': 0x1A, ']': 0x1B, 'enter': 0x1C, 'ctrl': 0x1D,
        'a': 0x1E, 's': 0x1F, 'd': 0x20, 'f': 0x21, 'g': 0x22, 'h': 0x23, 'j': 0x24, 'k': 0x25, 'l': 0x26, ';': 0x27, "'": 0x28, '`': 0x29, 'shift': 0x2A, '\\': 0x2B,
        'z': 0x2C, 'x': 0x2D, 'c': 0x2E, 'v': 0x2F, 'b': 0x30, 'n': 0x31, 'm': 0x32, ',': 0x33, '.': 0x34, '/': 0x35, 'shift_r': 0x36,
        'alt': 0x38, 'space': 0x39, 'caps': 0x3A,
        'f1': 0x3B, 'f2': 0x3C, 'f3': 0x3D, 'f4': 0x3E, 'f5': 0x3F, 'f6': 0x40, 'f7': 0x41, 'f8': 0x42, 'f9': 0x43, 'f10': 0x44,
        'num_lock': 0x45, 'scroll_lock': 0x46,
        'up': 0xC8, 'left': 0xCB, 'right': 0xCD, 'down': 0xD0,
        'insert': 0xD2, 'delete': 0xD3, 'home': 0xC7, 'end': 0xCF, 'pageup': 0xC9, 'pagedown': 0xD1,
        # 별칭(Alias) 처리
        'jump': 0x38,     # Alt (기본값)
        'attack': 0x1D,   # Ctrl (기본값)
        'loot': 0x2C,     # Z
        'interact': 0x39, # Space
    }

    def __init__(self, mode: str = "SOFTWARE", serial_port: str = None, physics_config_path: str = "physics_engine.json"):
        self.mode = mode.upper()
        # self.serial = None  <-- [삭제] ArduinoBridge로 대체
        self.arduino = None # [추가] 브리지 인스턴스
        self._stop_event = threading.Event()
        
        # 물리 엔진 설정 로드
        self.physics_data = self._load_physics_config(physics_config_path)
        self.tracker = MovementTracker()
        
        # 물리 상수 캐싱
        self.walk_acc = self.physics_data.get('walk_acceleration', 15.0)
        self.max_walk_speed = self.physics_data.get('max_walk_velocity', 125.0)
        
        # [수정] 하드웨어 모드 초기화 로직 변경
        if self.mode == "HARDWARE":
            if serial_port:
                logger.info(f"ActionHandler: Initializing Hardware Mode via {serial_port}...")
                self.arduino = ArduinoBridge(serial_port)
                # 연결 성공 여부 확인 (간단한 체크)
                if not self.arduino.serial:
                    logger.warning("Hardware initialization failed. Fallback to SOFTWARE.")
                    self.mode = "SOFTWARE"
            else:
                logger.warning("Serial Port not provided. Falling back to SOFTWARE mode.")
                self.mode = "SOFTWARE"
        
        if self.mode == "SOFTWARE":
            logger.info("ActionHandler: Software Mode (DirectInput Extended) Initialized.")

    def _load_physics_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Physics config not found. Using default values.")
            return {}

    # --- Low Level Input Methods ---

    def _get_scan_code(self, key_name: str) -> int:
        """키 이름을 스캔코드로 변환 (대소문자 무관)"""
        key = str(key_name).lower()
        return self.SCANCODE_MAP.get(key, 0)

    #@trace_logic
    def _send_key_software(self, key_code: int, pressed: bool):
        """DirectInput 방식 (SendInput)"""
        if key_code == 0: return # 매핑되지 않은 키 무시

        # [디버깅용 로그 추가] 현재 키 입력을 받고 있는 창의 제목 확인
        # (너무 자주 뜨면 주석 처리하세요)
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        logger.debug(f"[Input Target] '{buff.value}' (Key: {key_code})")

        extra = ctypes.c_ulong(0)
        ii_ = InputWrapper()
        
        flags = 0x0008  # KEYEVENTF_SCANCODE
        if not pressed:
            flags |= 0x0002  # KEYEVENTF_KEYUP

        ii_.ki = KeyBdInput(0, key_code, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    #def _send_key_hardware(self, key_name: str, pressed: bool):
    #    """Arduino Serial 통신 방식 (사용자 아두이노 코드 호환용)"""
    #    if not self.serial: return
        
        # 1. 아두이노가 기대하는 명령어: P(누름), R(뗌)
    #    cmd = "P" if pressed else "R"
        
        # 2. 아두이노가 기대하는 포맷: 쉼표 없이 붙여서 전송 (예: "Pleft\n")
        # 제공해주신 아두이노 코드는 '\n'을 만나야 명령을 처리하므로 개행문자 추가 필수
    #    payload = f"{cmd}{key_name}\n"
        
    #    try:
    #        self.serial.write(payload.encode())
    #    except Exception as e:
    #        logger.error(f"Serial Write Error: {e}")

    # [수정] 모드에 따른 분기 처리 및 키 매핑 적용
    @trace_logic
    def key_down(self, key_name: str):
        if self.mode == "SOFTWARE":
            code = self._get_scan_code(key_name)
            if code == 0: pass
            self._send_key_software(code, True)
        elif self.mode == "HARDWARE" and self.arduino:
            # [연결 지점] Bridge로 위임
            self.arduino.send_key(key_name, True)

    @trace_logic
    def key_up(self, key_name: str):
        if self.mode == "SOFTWARE":
            code = self._get_scan_code(key_name)
            self._send_key_software(code, False)
        elif self.mode == "HARDWARE" and self.arduino:
            # [연결 지점] Bridge로 위임
            self.arduino.send_key(key_name, False)

    def press(self, key_name: str, duration: float = 0.05):
        """단발성 키 입력 (HumanInput 적용)"""
        self.key_down(key_name)
        
        # [수정] 단순 sleep 대신 human_sleep 사용
        # duration은 '평균적으로 누르고 싶은 시간'이 됩니다.
        HumanInput.human_sleep(duration)
        
        self.key_up(key_name)
        
        # [수정] 키를 뗀 후 다음 행동까지의 미세한 딜레이 (After-cast delay)
        # 0.02초 ~ 0.05초 사이의 Ex-Gaussian 딜레이
        HumanInput.human_sleep(0.03)


    def mouse_move(self, x: int, y: int):
        """
        절대 좌표로 마우스 이동
        """
        # [HOOK] 레이턴시 측정 시작 트리거
        # 실제 하드웨어 전송 직전에 타임스탬프를 찍습니다.
        if self.mode == "HARDWARE" and latency_monitor.should_measure():
             latency_monitor.start_measurement()

        if self.mode == "HARDWARE" and self.arduino:
            self.arduino.send_mouse_move(x, y, 1920, 1080)
        else:
            logger.warning("Mouse move not implemented for SOFTWARE mode yet.")

    def mouse_down(self, button: str = 'left'):
        if self.mode == "HARDWARE" and self.arduino:
            # ArduinoBridge에 send_mouse_click(pressed=True)가 있다고 가정
            self.arduino.send_mouse_click(button, True)
        else:
            # SOFTWARE 모드 구현 (필요시 win32api 사용)
            pass

    # [추가] 수동 제어용: 마우스 버튼 뗌
    def mouse_up(self, button: str = 'left'):
        if self.mode == "HARDWARE" and self.arduino:
            self.arduino.send_mouse_click(button, False)
        else:
            pass

    # [추가] 마우스 클릭 기능 확장
    def mouse_click(self, button: str = 'left'):
        if self.mode == "HARDWARE" and self.arduino:
            self.arduino.send_mouse_click(button, True)
            HumanInput.human_sleep(0.05 + HumanInput.get_random_delay(0.01, 0.03)) # 클릭 지속시간 랜덤화
            self.arduino.send_mouse_click(button, False)
        else:
             # SOFTWARE 모드 클릭 미구현 시 로그
             pass

    # --- Physics & Feedback Control Logic (기존 유지) ---

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

    #@trace_logic
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
                if self.tracker.check_stuck(time.time(), threshold_dist=5, timeout=3.0):
                    logger.warning(f"Stuck detected at {curr_x}")
                    self.key_up(direction)
                    return self._recover_from_stuck(get_current_pos)

                HumanInput.human_sleep(0.05)

            return False
        finally:
            self.key_up(direction)
            if self.mode == "SOFTWARE" and self.max_walk_speed > 100: 
                opp_key = 'left' if direction == 'right' else 'right'
                # 관성 제어용 반대키 입력도 랜덤화 적용
                self.press(opp_key, 0.05)

    def _recover_from_stuck(self, get_current_pos: Callable) -> bool:
        self.press('jump')
        time.sleep(0.5)
        
        now_pos = get_current_pos()
        if now_pos and self.tracker.prev_pos:
            return PhysicsUtils.calc_distance(now_pos, self.tracker.prev_pos) > 10
        return False

    #@trace_logic
    def jump_shot(self, direction: Optional[str] = None, jump_key: str = 'jump', attack_key: str = 'attack'):
        if direction: self.key_down(direction)
        
        # [수정] 점프와 공격 사이의 간격 랜덤화
        self.press(jump_key, 0.1) 
        HumanInput.human_sleep(0.08) # 0.05 -> 0.08 (약간 늘리면서 랜덤화)
        
        self.press(attack_key, 0.1)
        
        if direction:
            HumanInput.human_sleep(0.1)
            self.key_up(direction)

    def emergency_stop(self):
        self._stop_event.set()
        # 모든 주요 키 Release 시도
        keys_to_release = ['left', 'right', 'up', 'down', 'alt', 'ctrl', 'shift', 'space']
        for key in keys_to_release:
            self.key_up(key)
        logger.info("EMERGENCY STOP executed.")