# ui/simulation_mode.py

import tkinter as tk
import time
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from engine.physics_engine import PhysicsEngine
from utils.logger import logger, trace_logic
from ui.components.viewport_manager import ViewportManager # [추가] 클래스 임포트
import math

class SimulationMode:
    """
    시뮬레이션 모드의 로직과 렌더링을 담당하는 클래스
    메인 윈도우의 캔버스를 공유하여 사용합니다.
    """
    def __init__(self, main_window):
        self.mw = main_window
        self.canvas = main_window.canvas
        
        # [수정] 메인 윈도우의 뷰포트를 공유하지 않고, 독립적인 인스턴스 생성
        self.viewport = ViewportManager() 
        
        # 1. 엔진 초기화
        self.map_processor = self.mw.agent.map_processor
        self.physics_engine = PhysicsEngine()
        
        if self.mw.cur_rf_path:
            self.physics_engine.load_model(self.mw.cur_rf_path)
        
        self.path_finder = PathFinder(self.map_processor, self.physics_engine)
        
        # 2. 캐릭터 상태 변수
        self.char_x = 125.0
        self.char_y = 66.0
        self.vx = 0.0
        self.vy = 0.0
        self.is_ground = True
        
        self.active = False
        self.last_time = time.time()
        
        # [수정] 맵 데이터에 맞춰 월드 크기 설정
        if self.map_processor.platforms:
            max_x = max(p['x_end'] for p in self.map_processor.platforms)
            max_y = max(p['y'] for p in self.map_processor.platforms)
            self.viewport.set_world_size(max_x + 20, max_y + 20)
        else:
            self.viewport.set_world_size(300, 200)
            
        # 시뮬레이션에 적합한 기본 줌 설정 (미니맵은 작으므로 크게 확대)
        self.viewport.zoom_scale = 4.0 
        self.viewport.center_view() # 뷰 중앙 정렬

    def start(self):
        self.active = True
        self.last_time = time.time()
        self.char_x = 125.0 # 초기 위치 리셋
        self.char_y = 66.0
        self.vx, self.vy = 0.0, 0.0
        print(">>> 시뮬레이션 모드 시작")

    def stop(self):
        self.active = False
        print(">>> 시뮬레이션 모드 종료")

    @trace_logic
    def update(self):
        """매 프레임 호출되어 물리 연산 및 화면 갱신"""
        if not self.active:
            return

        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        # 1. 물리 업데이트 (중력 적용)
        # 물리 엔진 모델이 로드되지 않았다면 기본 중력 사용
        gravity = 5.0 
        if self.physics_engine.is_loaded:
            # 여기서는 간단히 중력만 시뮬레이션 (액션 입력은 별도 함수로 처리)
            # 실제로는 현재 수행 중인 액션에 따라 predict_velocity를 호출해야 함
            pass 
            
        if not self.is_ground:
            self.vy += gravity * 0.1 # 간단한 중력 보정
            
        # 위치 업데이트
        self.char_x += self.vx
        self.char_y += self.vy
        
        # 2. 충돌 처리 (MapProcessor 활용)
        plat = self.map_processor.find_current_platform(self.char_x, self.char_y)
        
        if plat:
            # 떨어지다가 발판에 닿음 -> 착지
            if self.vy > 0 and abs(self.char_y - plat['y']) < 5:
                self.char_y = plat['y']
                self.vy = 0
                self.is_ground = True
        else:
            self.is_ground = False
            
        # 바닥으로 무한 추락 방지 (임시 바닥)
        if self.char_y > 300: 
            self.char_y = 66
            self.vy = 0
            self.is_ground = True

        # 마찰력 (지상 감속)
        if self.is_ground:
            self.vx *= 0.8
            if abs(self.vx) < 0.1: self.vx = 0

        # 3. 화면 그리기
        self.draw()

    @trace_logic
    def apply_action(self, action_name):
        """사용자 입력 또는 봇의 행동을 물리 엔진에 반영"""
        if not self.physics_engine.is_loaded:
            # 물리 엔진 없으면 하드코딩 값 사용 (Fallback)
            if action_name == "move_left": self.vx = -2
            elif action_name == "move_right": self.vx = 2
            elif action_name == "jump": self.vy = -5; self.is_ground = False
            elif action_name == "up_jump": self.vy = -10; self.is_ground = False
            return

        # 물리 엔진 모델 사용 (Action Index 매핑 필요)
        # 0: left, 1: right, 2: jump, 3: up, 4: down (모델 학습시 정의된 순서)
        act_map = {"move_left": 0, "move_right": 1, "jump": 2, "up_jump": 3, "down_jump": 4}
        idx = act_map.get(action_name, -1)
        
        if idx != -1:
            # 모델 예측: 현재 상태(지상여부)에서 해당 행동 시 속도 변화
            vel, gravity = self.physics_engine.predict_velocity(idx, self.is_ground)
            # 모델 출력(vel)은 [vx, vy] 형태라고 가정
            # 값을 적절히 스케일링하여 적용
            self.vx = float(vel[0]) 
            self.vy = float(vel[1]) 
            self.is_ground = False # 행동 시작 시 일단 공중으로 간주

    @trace_logic
    def draw(self):
        self.canvas.delete("sim_obj") 
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # [수정] 자신의 viewport 인스턴스 사용
        def to_screen(x, y):
            return self.viewport.world_to_screen(x, y, cw, ch)

        # 1. 발판 그리기
        for plat in self.map_processor.platforms:
            x1, y1 = to_screen(plat['x_start'], plat['y'])
            x2, y2 = to_screen(plat['x_end'], plat['y'])
            self.canvas.create_line(x1, y1, x2, y2, fill="#00FF00", width=3, tags="sim_obj")

        # 2. 캐릭터 그리기
        px, py = to_screen(self.char_x, self.char_y)
        self.canvas.create_oval(px-10, py-25, px+10, py, fill="#FF5555", outline="white", width=2, tags="sim_obj")
        
        # 3. 정보 텍스트 (자신의 줌 스케일 표시)
        info_text = f"Sim Pos: ({self.char_x:.1f}, {self.char_y:.1f}) | Zoom: x{self.viewport.zoom_scale:.1f}"
        self.canvas.create_text(10, 10, anchor="nw", text=info_text, fill="yellow", font=("Arial", 12, "bold"), tags="sim_obj")