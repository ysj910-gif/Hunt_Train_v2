#modules\scanner.py

import cv2
import numpy as np
import pytesseract
import time
from utils.logger import logger

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GameScanner:
    def __init__(self):
        self.minimap_roi = None
        self.kill_roi = None
        
        # 상태 캐싱
        self.last_player_pos = (0, 0)
        self.current_kill_count = 0
        self.last_ocr_time = 0
        
        # 스킬 상태 관리
        self.skill_configs = {} # {name: {'rect':..., 'threshold':...}}
        self.skill_status = {}  # {name: is_cooldown}

    def set_rois(self, minimap_rect, kill_rect):
        self.minimap_roi = minimap_rect
        self.kill_roi = kill_rect
        logger.debug(f"ROI 설정 완료 - 미니맵: {minimap_rect}, 킬카운트: {kill_rect}")

    def find_player(self, frame):
        """미니맵에서 플레이어(노란 점) 위치 찾기 - 기존 로직 복원"""
        if not self.minimap_roi:
            return 0, 0

        x, y, w, h = self.minimap_roi
        
        # [수정] 기존 코드의 엄격한 범위 체크 복원
        # ROI가 화면 밖으로 나가는 경우 방지
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0, 0

        minimap = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # 노란색 범위 (기존 값 유지)
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.last_player_pos = (cx, cy)
                return cx, cy
        
        # [수정] 못 찾았을 경우, 기존 코드처럼 0, 0 반환 (혹은 마지막 위치 반환)
        # 위치를 못 잡는 문제를 해결하기 위해 찾지 못하면 0,0을 리턴하여 재탐색 유도
        return 0, 0

    def read_kill_count(self, frame):
        """OCR로 킬 카운트 읽기 (0.5초 간격 제한)"""
        if not self.kill_roi:
            return self.current_kill_count
            
        if time.time() - self.last_ocr_time < 0.5:
            return self.current_kill_count
            
        self.last_ocr_time = time.time()
        
        try:
            x, y, w, h = self.kill_roi
            if x+w > frame.shape[1] or y+h > frame.shape[0]:
                return self.current_kill_count

            roi = frame[y:y+h, x:x+w]
            # OCR 인식률 향상을 위한 전처리
            roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_not(thresh)
            
            txt = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            
            if txt.strip().isdigit():
                val = int(txt.strip())
                if val != self.current_kill_count:
                    self.current_kill_count = val
                    # 킬 카운트 변화는 중요하므로 디버그 로그에 기록
                    # logger.debug(f"Kill Count Updated: {val}")
                    
        except Exception:
            pass # OCR 에러는 무시 (로그 과다 방지)
            
        return self.current_kill_count

    def register_skill(self, name, rect, frame=None):
        """스킬 쿨타임 감지 영역 등록 및 기준값 자동 설정"""
        threshold = 100.0 # 기본값
        
        if frame is not None:
            x, y, w, h = rect
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                active_v = np.mean(hsv[:, :, 2])
                threshold = active_v * 0.75 # 현재 밝기의 75%를 기준으로 설정
                logger.info(f"스킬 [{name}] 등록: 현재밝기({active_v:.1f}) -> 기준({threshold:.1f})")

        self.skill_configs[name] = {'rect': rect, 'threshold': threshold}

    def update_skill_status(self, frame):
        """모든 등록된 스킬의 쿨타임 여부 갱신"""
        for name, config in self.skill_configs.items():
            x, y, w, h = config['rect']
            thresh = config['threshold']
            
            if y+h > frame.shape[0] or x+w > frame.shape[1]:
                continue
                
            roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            current_v = np.mean(hsv[:, :, 2])
            
            # 어두우면(기준값 미만) 쿨타임 중(True)
            is_cooldown = current_v < thresh
            self.skill_status[name] = is_cooldown

    def is_cooldown(self, name):
        return self.skill_status.get(name, False)