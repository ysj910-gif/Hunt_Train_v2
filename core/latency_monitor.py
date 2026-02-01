# core/latency_monitor.py

import time
import cv2
import numpy as np
from utils.logger import logger

class LatencyMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LatencyMonitor, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.initialized = True
        
        # 설정
        self.measure_interval = 30.0  # 측정 주기 (초)
        self.visual_threshold = 500000  # 화면 변화 감지 임계값 (상황에 따라 조절 필요)
        
        # 상태 변수
        self.last_measure_time = 0
        self.start_time = 0
        self.is_measuring = False
        self.current_latency = 0.0 # ms 단위
        self.prev_frame_gray = None

    def should_measure(self):
        """측정 주기가 되었는지 확인"""
        return (time.time() - self.last_measure_time) > self.measure_interval

    def start_measurement(self):
        """명령 전송 시점 기록 (ActionHandler에서 호출)"""
        if self.is_measuring or not self.should_measure():
            return
        
        self.is_measuring = True
        self.start_time = time.time()
        # logger.debug("[Latency] 측정 시작: 명령 전송됨")

    def check_visual_change(self, frame):
        """화면 변화 감지 및 레이턴시 계산 (VisionSystem에서 호출)"""
        if not self.is_measuring or frame is None:
            self.prev_frame_gray = None
            return

        # 1. 흑백 변환 (연산 속도 최적화)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return

        # 2. 이전 프레임과 차이 계산 (Frame Difference)
        diff = cv2.absdiff(self.prev_frame_gray, gray)
        non_zero_count = np.sum(diff)  # 단순히 픽셀 차이의 합

        # 3. 임계값 이상이면 화면이 반응한 것으로 간주
        if non_zero_count > self.visual_threshold:
            end_time = time.time()
            self.current_latency = (end_time - self.start_time) * 1000 # ms 변환
            self.last_measure_time = end_time
            self.is_measuring = False
            self.prev_frame_gray = None # 리셋
            
            logger.info(f"📈 [Latency] 측정 완료: {self.current_latency:.1f} ms (Diff: {non_zero_count})")
        else:
            # 변화가 미미하면 현재 프레임 저장 후 다음 프레임 대기
            self.prev_frame_gray = gray

    def get_latency_info(self):
        """UI 표시용 정보 반환"""
        status = "Measuring..." if self.is_measuring else "Idle"
        return f"{self.current_latency:.0f}ms"

# 싱글톤 인스턴스
latency_monitor = LatencyMonitor()