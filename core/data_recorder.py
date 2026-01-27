# core/data_recorder.py

import csv
import time
import os
import datetime
import cv2
import numpy as np
from utils.logger import logger

class DataRecorder:
    def __init__(self, map_processor=None, filename_prefix="Physics_Record"):
        """
        :param map_processor: 지상/공중 여부를 판단하기 위한 MapProcessor 인스턴스 (필수)
        :param filename_prefix: 저장될 파일의 접두사
        """
        self.map_processor = map_processor
        
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"data/{filename_prefix}_{timestamp}.csv"
        
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        # --- [물리 학습을 위한 확장 헤더 설정] ---
        self.headers = [
            "timestamp", 
            "scenario",      # 실험 시나리오 (예: friction_test, jump_inertia)
            "state",         # 봇 상태 (PHYSICS_TEST, IDLE 등)
            "key_pressed",   # 입력 키 (action)
            
            # 위치 정보
            "player_x", "player_y",
            
            # 물리 정보 (속도, 가속도, 상태) - 핵심 피처
            "vx", "vy",          # 현재 속도 (pixels/sec)
            "ax", "ay",          # 현재 가속도 (pixels/sec^2)
            "is_ground",      # 바닥에 닿았는지
            "is_wall_left",   # 왼쪽 벽에 붙었는지
            "is_wall_right",  # 오른쪽 벽에 붙었는지
            "is_ladder",      # 사다리/줄에 매달렸는지 (완전 다른 물리 적용)
            "air_time"   # [추가] 공중에 떠 있는 시간 (초)
            
            # 기타 보조 정보
            "entropy",       # 이미지 복잡도
            "platform_id"    # 현재 밟고 있는 발판 ID (없으면 -1)
        ]
        self.writer.writerow(self.headers)
        
        # 물리 계산을 위한 이전 프레임 상태 저장 변수
        self.prev_time = None
        self.prev_x = None
        self.prev_y = None
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        
        # 현재 실험 시나리오 이름
        self.current_scenario = "None"

        logger.info(f"✅ 물리 데이터 레코더 시작: {self.filepath}")

    def set_scenario(self, scenario_name):
        """현재 진행 중인 실험 시나리오 이름을 설정합니다."""
        if self.current_scenario != scenario_name:
            logger.info(f"🧪 실험 시나리오 변경: {self.current_scenario} -> {scenario_name}")
            self.current_scenario = scenario_name

    def log_step(self, frame, player_pos, action, state):
        """
        매 프레임의 데이터를 물리 정보와 함께 CSV에 기록합니다.
        
        :param frame: 현재 화면 이미지 (CV2)
        :param player_pos: (x, y) 튜플
        :param action: 수행한 키 입력 (String)
        :param state: 현재 봇 상태
        """
        try:
            current_time = time.time()
            px, py = player_pos if player_pos else (0, 0)
            
            # 1. 물리 데이터 계산 (속도, 가속도)
            vx, vy = 0.0, 0.0
            ax, ay = 0.0, 0.0
            
            if self.prev_time is not None and self.prev_x is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    # 속도 계산 (v = dx / dt)
                    vx = (px - self.prev_x) / dt
                    vy = (py - self.prev_y) / dt
                    
                    # 가속도 계산 (a = dv / dt) - 노이즈가 있을 수 있으므로 참고용
                    ax = (vx - self.prev_vx) / dt
                    ay = (vy - self.prev_vy) / dt

            # 2. 지상/공중 상태 판별 (MapProcessor 활용)
            is_ground = 0
            is_wall_left = 0
            is_wall_right = 0
            is_ladder = 0
            platform_id = -1

            if self.map_processor:
                px, py = player_pos if player_pos else (0,0)
            
                # 1. 바닥 체크
                plat = self.map_processor.find_current_platform(px, py)
                if plat: is_ground = 1
                
                # 2. 벽 체크 (현재 발판의 양 끝점과 비교)
                # MapProcessor의 is_on_edge 활용 가능
                edge_status = self.map_processor.is_on_edge(px, py)
                if edge_status == 'left_edge':
                    is_wall_left = 1
                elif edge_status == 'right_edge':
                    is_wall_right = 1

            # [공중 체류 시간 계산]
            air_time = 0.0
            if is_ground == 0:
                if self.air_start_time is None:
                    self.air_start_time = current_time # 방금 떴음
                air_time = current_time - self.air_start_time
            else:
                self.air_start_time = None # 착지함 (리셋)
                    
           
            # 3. 이미지 엔트로피 (선택적)
            entropy = 0.0
            if frame is not None:
                # 연산 부하를 줄이기 위해 썸네일로 계산하거나 생략 가능
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5) # 크기 축소
                edges = cv2.Canny(small, 100, 200)
                entropy = np.sum(edges) / 255.0

            # 4. CSV 기록
            row = [
                f"{current_time:.4f}",
                self.current_scenario,
                state,
                action,
                
                px, py,
                
                f"{vx:.2f}", f"{vy:.2f}",
                f"{ax:.2f}", f"{ay:.2f}",
                is_ground, is_wall_left, is_wall_right, is_ladder,
                f"{air_time:.3f}",                
                f"{entropy:.2f}",
                platform_id
            ]
            self.writer.writerow(row)
            
            # 5. 상태 업데이트
            self.prev_time = current_time
            self.prev_x = px
            self.prev_y = py
            self.prev_vx = vx
            self.prev_vy = vy
            
        except Exception as e:
            # 기록 중 에러가 나도 봇이 멈추지 않도록 처리
            logger.error(f"Recording Error: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            logger.info("✅ 데이터 녹화 파일 저장 완료")