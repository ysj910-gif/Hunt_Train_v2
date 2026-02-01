#modules\scanner.py

import cv2
import numpy as np
import pytesseract
import time
from utils.logger import logger, trace_logic

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

        self.last_roi_log_time = 0
        self.last_minimap_roi = None
        self.last_kill_roi = None

    def set_rois(self, minimap_roi, kill_roi):
        """
        VisionSystem에서 인식한 ROI 정보를 받아옴 (스로틀링 적용됨)
        """
        self.minimap_roi = minimap_roi
        self.kill_roi = kill_roi

        # --- [핵심 수정] 로그 출력 조건 체크 ---
        should_log = False
        current_time = time.time()

        # 1. 값이 이전과 달라졌는지 확인 (새롭게 설정될 때)
        if (minimap_roi != self.last_minimap_roi) or (kill_roi != self.last_kill_roi):
            should_log = True
        
        # 2. 마지막 기록 후 60초가 지났는지 확인 (주기적 생존 신고)
        if (current_time - self.last_roi_log_time) > 60:
            should_log = True

        # 조건이 맞을 때만 로그 출력 및 상태 업데이트
        if should_log:
            from utils.logger import logger # 필요시 import
            logger.debug(f"ROI 설정 완료 - 미니맵: {minimap_roi}, 킬카운트: {kill_roi}")
            
            self.last_minimap_roi = minimap_roi
            self.last_kill_roi = kill_roi
            self.last_roi_log_time = current_time

    @trace_logic
    def find_player(self, frame):
        """
        미니맵에서 플레이어(노란 점) 위치 찾기 - 서브픽셀 정밀도 적용
        """
        if not self.minimap_roi:
            return 0, 0

        x, y, w, h = self.minimap_roi
        
        # 화면 범위 체크
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0, 0

        minimap = frame[y:y+h, x:x+w]
        
        # [핵심 수정] 서브픽셀 인식을 위한 업스케일링 (4배 확대)
        scale_factor = 4.0 
        
        # 보간법: CUBIC이나 LINEAR를 써야 경계면이 부드러워져 무게중심이 정밀해짐
        upscaled = cv2.resize(minimap, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
        
        # 노란색 범위 (기존 값 유지)
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                # [수정] int()로 자르지 않고 float 상태로 계산 후 스케일로 나눔
                # 결과적으로 (12.5, 30.25) 같은 정밀 좌표를 얻음
                cx = (M["m10"] / M["m00"]) / scale_factor
                cy = (M["m01"] / M["m00"]) / scale_factor
                
                self.last_player_pos = (cx, cy)
                return cx, cy
        
        # 못 찾으면 0, 0 반환
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
                    logger.debug(f"Kill Count Updated: {val}")
                    
        except Exception:
            pass # OCR 에러는 무시 (로그 과다 방지)
            
        return self.current_kill_count

    @trace_logic
    def register_skill(self, name, rect, frame=None, threshold=None): # [수정] threshold 파라미터 추가
        """스킬 쿨타임 감지 영역 등록"""
        
        # 1. 임계값 설정 로직 개선
        final_threshold = 100.0 # 기본값
        
        if threshold is not None:
            final_threshold = threshold
        elif frame is not None:
            # 프레임이 있으면 자동 계산
            x, y, w, h = rect
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                active_v = np.mean(hsv[:, :, 2])
                final_threshold = active_v * 0.75 
                logger.info(f"스킬 [{name}] 자동 등록: 밝기({active_v:.1f}) -> 기준({final_threshold:.1f})")

        self.skill_configs[name] = {'rect': rect, 'threshold': final_threshold}
        
        # [중요] 등록 즉시 '사용 가능(False)' 상태로 초기화하여 봇이 인식하게 함
        self.skill_status[name] = False

    @trace_logic
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