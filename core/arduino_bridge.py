# core/arduino_bridge.py

import serial
import time
import threading
from utils.logger import logger

class ArduinoBridge:
    """
    PC -> Arduino 시리얼 통신을 담당하는 어댑터 클래스
    arduino.py의 프로토콜(A, P, R, B 커맨드)을 구현합니다.
    """
    
    # arduino.py의 특수 키 매핑 규칙 적용
    KEY_MAP = {
        'backspace': '\x08',
        'caps': 'caps_lock',
        'esc': 'esc',
        # 필요한 경우 추가 매핑
    }

    def __init__(self, port: str, baudrate: int = 500000):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.lock = threading.Lock() # 스레드 안전성 확보
        self._connect()

    def _connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.01)
            logger.info(f"ArduinoBridge: Connected to {self.port} at {self.baudrate}")
        except Exception as e:
            logger.error(f"ArduinoBridge Connection Error: {e}")
            self.serial = None

    def _send(self, command: str):
        if not self.serial or not self.serial.is_open:
            return
        
        try:
            with self.lock:
                # 아두이노 프로토콜은 끝에 '\n' 필수
                payload = f"{command}\n".encode()
                self.serial.write(payload)
                
                # 상세 로그 (디버깅용)
                if command.startswith('A'): # 마우스 이동은 너무 빈번하므로 제외하거나 DEBUG 레벨로
                    # logger.debug(f"[Arduino] Mouse Move: {command}")
                    pass
                else:
                    logger.info(f"[Arduino] Send: {command}")
                    
        except Exception as e:
            logger.error(f"ArduinoBridge Write Error: {e}")
            # 재연결 시도 로직이 필요하다면 여기에 추가

    def send_key(self, key_name: str, pressed: bool):
        """
        키보드 입력 전송
        P{key} : 누름
        R{key} : 뗌
        """
        # 매핑된 키가 있으면 변환, 없으면 그대로 사용 (소문자 처리)
        key = self.KEY_MAP.get(key_name.lower(), key_name.lower())
        
        cmd_prefix = "P" if pressed else "R"
        self._send(f"{cmd_prefix}{key}")

    def send_mouse_move(self, x: int, y: int, screen_w: int = 1920, screen_h: int = 1080):
        """
        마우스 절대 좌표 이동
        arduino.py 로직: 0~32767 범위로 정규화하여 전송
        프로토콜: A{x},{y}
        """
        # 범위 제한
        safe_x = max(0, min(x, screen_w))
        safe_y = max(0, min(y, screen_h))

        # 0 ~ 32767 로 변환
        abs_x = int((safe_x / screen_w) * 32767)
        abs_y = int((safe_y / screen_h) * 32767)
        
        self._send(f"A{abs_x},{abs_y}")

    def send_mouse_click(self, button: str, pressed: bool):
        """
        마우스 클릭
        button: 'left', 'right'
        프로토콜: BL1, BL0, BR1, BR0
        """
        btn_char = 'L' if button == 'left' else 'R'
        state = '1' if pressed else '0'
        self._send(f"B{btn_char}{state}")