# verify_physics.py

import tkinter as tk
import time
import math
import torch
import numpy as np
from engine.map_processor import MapProcessor
from engine.physics_engine import PhysicsEngine

# --- 설정 ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
SCALE_FACTOR = 0.12  # [중요] advanced_pathfinder.py에 적용한 스케일과 동일해야 함
VIRTUAL_MAP_H = 84   # 미니맵 높이 (가정)
VIRTUAL_MAP_W = 180  # 미니맵 너비 (가정)

# 화면 표시 배율 (너무 작아서 안 보일 수 있으니 3배 확대해서 보여줌)
VIEW_ZOOM = 3.0 

class PhysicsSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Physics Engine Simulator (Scale Verification)")
        
        # 캔버스 생성
        self.canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="black")
        self.canvas.pack()
        
        # 물리 엔진 로드
        self.physics = PhysicsEngine()
        self.use_ai_physics = self.physics.load_model("physics_hybrid_model.pth")
        
        if self.use_ai_physics:
            print("✅ AI 물리 모델 로드 성공 (HybridPhysicsNet)")
        else:
            print("⚠️ AI 모델 없음 -> 하드코딩된 물리(Fallback) 사용")

        # 초기 상태
        self.pos_x = VIRTUAL_MAP_W / 2
        self.pos_y = VIRTUAL_MAP_H - 10 # 바닥 근처
        self.vx = 0.0
        self.vy = 0.0
        self.gravity = 5.0 * SCALE_FACTOR
        
        # 키 입력 상태
        self.keys = set()
        
        # UI 요소
        self.player_id = self.canvas.create_rectangle(0, 0, 10, 10, fill="cyan", outline="white")
        self.floor_id = self.canvas.create_line(0, 0, WINDOW_WIDTH, 0, fill="green", width=2)
        self.info_text = self.canvas.create_text(10, 10, anchor="nw", fill="white", text="Loading...")
        
        # 이벤트 바인딩
        root.bind("<KeyPress>", self.on_key_down)
        root.bind("<KeyRelease>", self.on_key_up)
        
        # 루프 시작
        self.last_time = time.time()
        self.update()

    def on_key_down(self, event):
        self.keys.add(event.keysym)

    def on_key_up(self, event):
        if event.keysym in self.keys:
            self.keys.remove(event.keysym)

    def get_current_action_id(self):
        """키 입력을 물리 엔진용 Action ID로 변환"""
        # 매핑: {"move_left": 0, "move_right": 1, "jump": 2, "up_jump": 3, "down_jump": 4}
        
        up = 'Up' in self.keys
        down = 'Down' in self.keys
        left = 'Left' in self.keys
        right = 'Right' in self.keys
        jump = 'Alt_L' in self.keys or 'space' in self.keys # Alt 또는 Space
        
        action_name = "stand"
        act_id = -1
        
        if jump:
            if up: 
                act_id = 3; action_name = "up_jump"
            elif down: 
                act_id = 4; action_name = "down_jump"
            else: 
                act_id = 2; action_name = "jump"
        elif left:
            act_id = 0; action_name = "move_left"
        elif right:
            act_id = 1; action_name = "move_right"
            
        return act_id, action_name

    def update(self):
        # 1. Delta Time (단순화를 위해 고정 프레임처럼 처리하거나 실제 시간 반영)
        # 봇의 시뮬레이션은 보통 1프레임 단위로 계산하므로 여기도 단순 연산
        
        # 2. 행동 결정
        act_id, act_name = self.get_current_action_id()
        
        # 3. 물리 연산 (AStarPathFinder의 로직과 동일하게 구현)
        pred_vx, pred_vy = 0, 0
        
        if self.use_ai_physics and act_id != -1:
            # AI 모델 사용 (주의: 모델이 학습된 스케일이 원본 게임 기준이라면 여기서도 SCALE을 곱해줘야 함)
            # 여기서는 모델이 '실제 게임 픽셀'을 내뱉는다고 가정하고 스케일링 적용
            is_grounded = 1.0 if self.pos_y >= VIRTUAL_MAP_H - 5 else 0.0
            (pvx, pvy), pg = self.physics.predict_velocity(act_id, is_grounded)
            
            pred_vx = pvx * SCALE_FACTOR
            pred_vy = pvy * SCALE_FACTOR
            self.gravity = pg * SCALE_FACTOR
            
        elif act_id != -1:
            # Fallback (하드코딩) 물리 - advanced_pathfinder.py 수정본과 동일
            if act_name == "jump": pred_vy = -65.0 * SCALE_FACTOR
            if act_name == "up_jump": pred_vy = -140.0 * SCALE_FACTOR
            if act_name == "move_left": pred_vx = -18.0 * SCALE_FACTOR
            if act_name == "move_right": pred_vx = 18.0 * SCALE_FACTOR
            if act_name == "down_jump": pred_vy = -10.0 * SCALE_FACTOR
            
        # 4. 속도 적용 (관성 구현은 제외하고 즉시 속도 적용 - 봇 시뮬레이션 방식)
        # 키를 안 누르면 속도 0 (공중 이동 제외)
        if act_id == -1:
            if self.pos_y < VIRTUAL_MAP_H - 5: # 공중이면 좌우 관성 유지? 일단 0으로 둠
                self.vx = 0
            else:
                self.vx = 0
                self.vy = 0
        else:
            # 점프류는 '순간 힘'이므로 현재 속도에 더하거나 덮어씌움
            # 이동류는 '지속 속도'
            if "jump" in act_name:
                # 점프는 바닥에 있을 때만 (단, 윗점프는 로프 등 고려해야 하나 여기선 단순화)
                if self.pos_y >= VIRTUAL_MAP_H - 5 or act_name=="up_jump":
                    self.vy = pred_vy
                    self.vx = pred_vx # 점프 시 좌우 이동도 포함될 수 있음
            else:
                self.vx = pred_vx
                # self.vy는 건드리지 않음 (중력 영향)

        # 중력 적용
        if self.pos_y < VIRTUAL_MAP_H - 5: # 공중
            self.vy += self.gravity
        else:
            # 바닥에 닿음
            if self.vy > 0: self.vy = 0
            # 바닥 보정
            self.pos_y = VIRTUAL_MAP_H - 5

        # 5. 위치 업데이트
        self.pos_x += self.vx
        self.pos_y += self.vy
        
        # 6. 화면 그리기 (Zoom 적용)
        screen_x = self.pos_x * VIEW_ZOOM
        screen_y = self.pos_y * VIEW_ZOOM
        
        # 바닥 라인 그리기
        ground_y = (VIRTUAL_MAP_H - 5) * VIEW_ZOOM
        self.canvas.coords(self.floor_id, 0, ground_y, WINDOW_WIDTH, ground_y)
        
        # 플레이어 그리기
        size = 5 * VIEW_ZOOM
        self.canvas.coords(self.player_id, 
                           screen_x - size, screen_y - size*2, 
                           screen_x + size, screen_y)
        
        # 정보 텍스트
        info = f"Action: {act_name}\n"
        info += f"Pos: ({self.pos_x:.1f}, {self.pos_y:.1f})\n"
        info += f"Vel: ({self.vx:.1f}, {self.vy:.1f})\n"
        info += f"Scale: {SCALE_FACTOR} (Zoom: {VIEW_ZOOM}x)"
        self.canvas.itemconfig(self.info_text, text=info)
        
        # 다음 프레임 (60 FPS)
        self.root.after(16, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    sim = PhysicsSimulator(root)
    root.mainloop()