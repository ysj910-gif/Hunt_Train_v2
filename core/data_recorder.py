# core/data_recorder.py

import csv
import time
import os
import datetime
import cv2
import numpy as np
from utils.logger import logger
from core.latency_monitor import latency_monitor

class DataRecorder:
    def __init__(self, map_processor=None, filename_prefix="Physics_Record"):
        self.map_processor = map_processor
        
        if not os.path.exists("data"):
            os.makedirs("data")

        # 변수 초기화
        self.file = None
        self.writer = None
        self.last_filepath = None 
        
        # 물리 상태 변수 초기화
        self.prev_time = None
        self.prev_x = None
        self.prev_y = None
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.current_scenario = "None"
        self.air_start_time = None

        # [핵심] 중복 코드를 제거하고 open() 메서드 재사용
        # 이렇게 하면 __init__ 호출 시에도 파일이 열리므로 기존 코드와 호환됩니다.
        self.open(filename_prefix) 

    def open(self, filename_prefix):
        # 기존 파일 닫기 (self.file이 None이어도 안전하게 처리되도록 close 구현 필요)
        self.close()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"data/{filename_prefix}_{timestamp}.csv"
        self.last_filepath = self.filepath # 생성된 파일을 마지막 파일로 기록
        
        try:
            self.file = open(self.filepath, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            
            # 헤더 설정 (이곳 한 군데에서만 관리하면 됨)
            self.headers = [
                "timestamp", "scenario", "state", "key_pressed",
                "player_x", "player_y",
                "vx", "vy", "ax", "ay",
                "is_ground", "is_wall_left", "is_wall_right", "is_ladder", "air_time",
                "entropy", "platform_id", "latency"  # <--- 추가됨
            ]
            self.writer.writerow(self.headers)
            logger.info(f"✅ 데이터 기록 시작: {self.filepath}")
            
        except Exception as e:
            logger.error(f"파일 열기 실패: {e}")
            self.file = None

    def close(self):
        # self.file이 존재하는지 확인 후 닫기
        if hasattr(self, 'file') and self.file:
            try:
                self.file.close()
            except: pass
        self.file = None
        self.writer = None

    def set_scenario(self, scenario_name):
        """현재 진행 중인 실험 시나리오 이름을 설정합니다."""
        if self.current_scenario != scenario_name:
            logger.info(f"🧪 실험 시나리오 변경: {self.current_scenario} -> {scenario_name}")
            self.current_scenario = scenario_name

    def log_step(self, frame, player_pos, action, state, skill_status=None):
        """
        매 프레임의 데이터를 물리 정보와 함께 CSV에 기록합니다.
        
        :param frame: 현재 화면 이미지 (CV2)
        :param player_pos: (x, y) 튜플
        :param action: 수행한 키 입력 (String)
        :param state: 현재 봇 상태
        """
        if not self.file or not self.writer:
            return
        
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

            # [신규] 레이턴시 값 가져오기
            current_latency = latency_monitor.current_latency

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
                platform_id,
                f"{current_latency:.1f}"
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
