# core/data_recorder.py

import csv
import time
import os
import datetime
import cv2
import numpy as np
from utils.logger import logger

class DataRecorder:
    def __init__(self, filename_prefix="Record"):
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"data/{filename_prefix}_{timestamp}.csv"
        
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        # 헤더 설정
        self.headers = [
            "timestamp", 
            "entropy",       # 이미지 복잡도 (전투 상황 판단용)
            "state",         # 봇 상태 (COMBAT, IDLE 등)
            "action",        # 수행한 행동
            "player_x", "player_y",
            "kill_count"
        ]
        self.writer.writerow(self.headers)
        logger.info(f"✅ 데이터 레코더 시작: {self.filepath}")

    def log_step(self, frame, player_pos, action, state, skill_status=None):
        """매 프레임의 데이터를 CSV에 기록"""
        try:
            # 1. 엔트로피(이미지 복잡도) 계산
            entropy = 0.0
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                entropy = np.sum(edges) / 255.0

            # 2. 위치 정보 분해
            px, py = player_pos if player_pos else (0, 0)
            
            # 3. CSV 기록
            row = [
                time.time(),
                f"{entropy:.2f}",
                state,
                action,
                px, py,
                0 # kill_count는 scanner 객체에서 가져와야 하는데, 여기선 생략되거나 파라미터로 받아야 함
            ]
            self.writer.writerow(row)
            
        except Exception as e:
            logger.error(f"Recording Error: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            logger.info("✅ 데이터 녹화 파일 저장 완료")