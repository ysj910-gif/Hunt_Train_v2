import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog # [추가] 입력 팝업용
import threading
import time
import sys
import os
import json
import traceback
import logging

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [필수 모듈 임포트]
try:
    from core.bot_agent import BotAgent
    from utils.logger import logger
    from modules.vision_system import VisionSystem
    from modules.scanner import GameScanner
    from core.action_handler import ActionHandler
    from engine.map_processor import MapProcessor
    from engine.path_finder import PathFinder
    from engine.physics_engine import PhysicsEngine
    from core.decision_maker import DecisionMaker  
    from core.data_recorder import DataRecorder
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")

class MapVisualizer(tk.Canvas):
    """맵 구조와 캐릭터 위치를 시각화하는 위젯 (변경 없음)"""
    def __init__(self, master, width=600, height=300, bg="white"):
        super().__init__(master, width=width, height=height, bg=bg)
        self.map_data = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.padding = 50
        self.player_id = None
        self.target_id = None

    def load_map(self, map_data):
        self.delete("all")
        self.map_data = map_data
        self._calculate_scale()
        self._draw_static_objects()

    def _calculate_scale(self):
        platforms = self.map_data.get("platforms", [])
        if not platforms: return

        min_x = min(p['x_start'] for p in platforms)
        max_x = max(p['x_end'] for p in platforms)
        min_y = min(p['y'] for p in platforms)
        max_y = max(p['y'] for p in platforms)
        
        for r in self.map_data.get("ropes", []):
            ry = r.get('y')
            rh = r.get('h')
            if ry is not None and rh is not None:
                min_y = min(min_y, ry)
                max_y = max(max_y, ry + rh)

        map_w = max_x - min_x
        map_h = max_y - min_y
        
        if map_w == 0: map_w = 1
        if map_h == 0: map_h = 1

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        
        self.scale_x = (canvas_w - self.padding * 2) / map_w
        self.scale_y = (canvas_h - self.padding * 2) / map_h
        
        scale = min(self.scale_x, self.scale_y)
        self.scale_x = scale
        self.scale_y = scale

        self.offset_x = min_x
        self.offset_y = min_y

    def _to_canvas(self, x, y):
        cx = (x - self.offset_x) * self.scale_x + self.padding
        cy = (y - self.offset_y) * self.scale_y + self.padding
        return cx, cy

    def _draw_static_objects(self):
        for p in self.map_data.get("platforms", []):
            x1, y1 = self._to_canvas(p['x_start'], p['y'])
            x2, y2 = self._to_canvas(p['x_end'], p['y'])
            self.create_line(x1, y1, x2, y2, fill="black", width=3)
            
        for r in self.map_data.get("ropes", []):
            rx = r.get('x')
            ry = r.get('y')
            rh = r.get('h')
            if rx is not None and ry is not None and rh is not None:
                x, y = self._to_canvas(rx, ry)
                _, y_end = self._to_canvas(rx, ry + rh)
                self.create_line(x, y, x, y_end, fill="brown", width=2)

    def update_player(self, px, py):
        cx, cy = self._to_canvas(px, py)
        r = 5
        if self.player_id:
            self.coords(self.player_id, cx-r, cy-r, cx+r, cy+r)
        else:
            self.player_id = self.create_oval(cx-r, cy-r, cx+r, cy+r, fill="red", outline="red")
            
    def set_target(self, x_start, x_end, y):
        if self.target_id:
            self.delete(self.target_id)
            
        x1, y1 = self._to_canvas(x_start, y - 5)
        x2, y2 = self._to_canvas(x_end, y + 5)
        self.target_id = self.create_rectangle(x1, y1, x2, y2, outline="green", width=3, dash=(4, 2))


class PhysicsTrainerVisualApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍁 Maple Physics Trainer (Repeat Control)")
        self.root.geometry("1150x750")
        
        self.agent = None
        self.is_running = False
        self.map_path = tk.StringVar()
        self.instruction_text = tk.StringVar(value="맵 파일을 로드해주세요.")
        
        # [수정] 미션 목록 (ID, 이름, 기본 반복 횟수)
        self.missions_data = [
            ("M1", "마찰력 테스트 (우)", 5),
            ("M2", "마찰력 테스트 (좌)", 5),
            ("M3", "점프 테스트 (중앙)", 20),
            ("M4", "점프 테스트 (좌측 끝)", 20),
            ("M5", "점프 테스트 (우측 끝)", 20),
            ("M6", "관성 테스트 (더블 점프)", 30),
            ("M7", "러닝 점프 (단일 점프)", 30),
            ("M8", "자유 낙하 (발판 이탈)", 15),
            ("M9", "하향 점프 (Down Jump)", 15),
            ("M10", "급정거/방향전환", 15),
            ("M11", "공중 역추진 (Air Brake)", 15),  # 점프 중 반대키 입력
            ("M12", "공격 관성 (이동 중 공격)", 15), # 이동 중 공격 키 입력
            ("M13", "로프 매달리기/이동", 10),       # 로프 물리 확인
            ("M14", "로프 이탈 점프", 10)            # 로프에서 점프
                ]
        
        # 횟수 관리용 딕셔너리
        self.mission_reps = {mid: default_reps for mid, _, default_reps in self.missions_data}
        
        self._setup_ui()
        
    def _setup_ui(self):
        # 1. 상단 설정
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")
        ttk.Label(top_frame, text="맵 파일:").pack(side="left")
        ttk.Entry(top_frame, textvariable=self.map_path, width=40).pack(side="left", padx=5)
        ttk.Button(top_frame, text="📂 열기", command=self._browse_map).pack(side="left")
        ttk.Button(top_frame, text="🛑 중지", command=self._stop_training).pack(side="right")

        # 2. 중단 (맵 + 체크리스트)
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        map_frame = ttk.LabelFrame(mid_frame, text="📍 실시간 맵 & 위치")
        map_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.canvas = MapVisualizer(map_frame, bg="#f5f5f5")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # [수정] 리스트 프레임 너비 증가
        list_frame = ttk.LabelFrame(mid_frame, text="📋 To-Do Check List (Double-click to Edit)", width=320)
        list_frame.pack(side="right", fill="y")
        
        # [수정] 컬럼에 'reps' 추가
        cols = ("status", "name", "reps")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=20)
        
        self.tree.heading("status", text="상태")
        self.tree.heading("name", text="미션 내용")
        self.tree.heading("reps", text="반복")
        
        self.tree.column("status", width=50, anchor="center")
        self.tree.column("name", width=180)
        self.tree.column("reps", width=50, anchor="center")
        
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 더블 클릭 이벤트 바인딩 (횟수 수정)
        self.tree.bind("<Double-1>", self._on_tree_double_click)
        
        # 초기 데이터 삽입
        for mid, name, reps in self.missions_data:
            self.tree.insert("", "end", iid=mid, values=("⬜", name, f"{reps}회"))

        # 3. 하단
        bottom_frame = ttk.LabelFrame(self.root, text="📢 코치 지시사항", padding=10)
        bottom_frame.pack(fill="x", padx=10, pady=10)
        
        self.lbl_instruction = ttk.Label(bottom_frame, textvariable=self.instruction_text, 
                                         font=("Helvetica", 16, "bold"), foreground="blue", anchor="center")
        self.lbl_instruction.pack(fill="x", pady=(0, 10))
        
        self.btn_start = ttk.Button(bottom_frame, text="🚀 훈련 시작", command=self._start_training, state="disabled")
        self.btn_start.pack(fill="x", ipady=5)

    # [추가] 리스트 더블 클릭 시 횟수 수정
    def _on_tree_double_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id: return
        
        # 현재 설정된 값 가져오기
        current_reps = self.mission_reps.get(item_id, 10)
        mission_name = next((m[1] for m in self.missions_data if m[0] == item_id), "")

        # 입력 팝업
        new_reps = simpledialog.askinteger("반복 횟수 설정", 
                                         f"'{mission_name}'\n반복 횟수를 입력하세요:",
                                         parent=self.root,
                                         minvalue=1, maxvalue=100,
                                         initialvalue=current_reps)
        
        if new_reps:
            self.mission_reps[item_id] = new_reps
            # 트리뷰 업데이트 (상태 아이콘 유지를 위해 기존 값 읽기)
            current_values = self.tree.item(item_id, "values")
            self.tree.item(item_id, values=(current_values[0], current_values[1], f"{new_reps}회"))
            print(f"✅ {mission_name}: {new_reps}회로 변경됨.")

    def _update_mission_status(self, mission_id, status):
        def _update():
            icon = "⬜"
            if status == "active": icon = "🏃"
            elif status == "done": icon = "✅"
            
            # 현재 횟수 표시 유지
            reps = self.mission_reps.get(mission_id, 0)
            name = next((m[1] for m in self.missions_data if m[0] == mission_id), "")
            
            if name:
                self.tree.item(mission_id, values=(icon, name, f"{reps}회"))
                if status == "active":
                    self.tree.selection_set(mission_id)
                    self.tree.see(mission_id)
        self.root.after(0, _update)

    def _browse_map(self):
        path = filedialog.askopenfilename(title="맵 데이터 선택", filetypes=[("JSON Map", "*.json"), ("All Files", "*.*")])
        if path:
            self.map_path.set(path)
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                
                if "platforms" not in data:
                    raise ValueError("JSON 파일에 'platforms' 데이터가 없습니다.")

                self.root.update() 
                self.canvas.load_map(data)
                self.btn_start.config(state="normal")
                self.instruction_text.set(f"맵 로드 완료!\n리스트를 더블클릭하여 반복 횟수를 조정하세요.")
                logger.info(f"Map Loaded: {path}")
                
            except Exception as e:
                print("❌ 맵 로드 에러:")
                traceback.print_exc()
                messagebox.showerror("맵 로드 실패", f"오류: {e}")

    def _start_training(self):
        self.is_running = True
        self.btn_start.config(state="disabled")
        t = threading.Thread(target=self._training_routine)
        t.daemon = True
        t.start()

    def _stop_training(self):
        self.is_running = False
        if self.agent:
            self.agent.stop()
        self.instruction_text.set("훈련이 중지되었습니다.")
        self.btn_start.config(state="normal")

    def _update_gui(self, text, target_area=None):
        self.instruction_text.set(text)
        if target_area:
            self.canvas.set_target(*target_area)
        else:
            if self.canvas.target_id:
                self.canvas.delete(self.canvas.target_id)
                self.canvas.target_id = None

    def _get_player_pos(self):
        if not self.agent or not self.agent.scanner:
            return None
        if hasattr(self.agent.scanner, 'player_pos'): return self.agent.scanner.player_pos
        if hasattr(self.agent.scanner, 'pos'): return self.agent.scanner.pos
        if hasattr(self.agent.scanner, 'data') and isinstance(self.agent.scanner.data, dict):
            return self.agent.scanner.data.get('pos')
        return None

    def _training_routine(self):
        try:
            print(">>> [INIT] Modules assembly...")
            
            vision_system = VisionSystem()
            scanner = GameScanner()
            action_handler = ActionHandler() 
            map_processor = MapProcessor()
            physics_engine = PhysicsEngine()
            
            if os.path.exists("physics_hybrid_model.pth"):
                physics_engine.load_model("physics_hybrid_model.pth")
            
            path_finder = PathFinder(map_processor, physics_engine)
            recorder = DataRecorder("Session_Log")
            
            self.agent = BotAgent(
                vision=vision_system,
                scanner=scanner,
                action_handler=action_handler,
                map_processor=map_processor,
                path_finder=path_finder,
                recorder=recorder
            )
            
            self.agent.is_recording = False
            brain = DecisionMaker(self.agent)
            self.agent.set_brain(brain)
            
            print(">>> [INIT] BotAgent ready.")

            if not self.agent.map_processor.load_map(self.map_path.get()):
                self._update_gui("맵 데이터 로드 실패!")
                return

            t_agent = threading.Thread(target=self.agent.start)
            t_agent.daemon = True
            t_agent.start()

            self.root.after(100, self._visualizer_loop)

            platforms = self.agent.map_processor.platforms
            if not platforms: raise ValueError("No platforms found.")

            run_plat = max(platforms, key=lambda p: p['x_end'] - p['x_start'])
            jump_plats = sorted(platforms, key=lambda p: p['y'])
            main_jump_plat = jump_plats[0] if jump_plats else run_plat

            # === [훈련 시작] ===
            self._update_gui("⚠️ 훈련 세션이 시작되었습니다.")
            time.sleep(2)

            # [수정] 각 미션마다 self.mission_reps에서 횟수를 가져와 실행
            
            # 1. 마찰력 (우)
            mid = "M1"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "middle", "마찰력 테스트 (우)\n중앙으로 이동하세요.")
            self._mission_action("Friction_Right", "오른쪽으로 3초간 달리고 멈추세요!", 
                                 duration=3.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 2. 마찰력 (좌)
            mid = "M2"
            self._update_mission_status(mid, "active")
            self._mission_action("Friction_Left", "왼쪽으로 3초간 달리고 멈추세요!", 
                                 duration=3.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 3. 점프 (중앙)
            mid = "M3"
            self._update_mission_status(mid, "active")
            self._mission_move_to(main_jump_plat, "middle", "점프 테스트 (중앙)\n가장 높은 발판으로 이동하세요.")
            self._mission_action("Jump_Middle_Neutral", "제자리 점프 (방향키 X)", 
                                 repeat=self.mission_reps[mid], wait=1.5)
            self._update_mission_status(mid, "done")
            
            # 4. 점프 (좌측)
            mid = "M4"
            self._update_mission_status(mid, "active")
            self._mission_move_to(main_jump_plat, "left_edge", "발판 '왼쪽 끝'으로 이동하세요.")
            self._mission_action("Jump_LeftEdge_Neutral", "왼쪽 끝에서 제자리 점프", 
                                 repeat=self.mission_reps[mid], wait=1.5)
            self._update_mission_status(mid, "done")

            # 5. 점프 (우측)
            mid = "M5"
            self._update_mission_status(mid, "active")
            self._mission_move_to(main_jump_plat, "right_edge", "발판 '오른쪽 끝'으로 이동하세요.")
            self._mission_action("Jump_RightEdge_Neutral", "오른쪽 끝에서 제자리 점프", 
                                 repeat=self.mission_reps[mid], wait=1.5)
            self._update_mission_status(mid, "done")

            # 6. 관성 (더블 점프)
            mid = "M6"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "관성 테스트 (더블 점프)\n긴 발판 왼쪽 끝으로 이동하세요.")
            self._mission_action("DoubleJump_Right", "오른쪽으로 달리면서 더블 점프!", 
                                 duration=3.0, wait=2.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 7. 러닝 점프
            mid = "M7"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "러닝 점프 테스트 (단일)\n긴 발판 왼쪽 끝으로 이동하세요.")
            self._mission_action("RunJump_Right", "오른쪽으로 달리면서 점프! (더블점프 금지)", 
                                 duration=3.0, wait=2.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 8. 자유 낙하
            mid = "M8"
            self._update_mission_status(mid, "active")
            self._mission_move_to(main_jump_plat, "left_edge", "자유 낙하 테스트\n높은 발판 왼쪽 끝으로 이동하세요.")
            self._mission_action("FreeFall_Left", "왼쪽으로 걸어서 발판 아래로 떨어지세요!", 
                                 duration=2.0, wait=3.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 9. 하향 점프
            mid = "M9"
            self._update_mission_status(mid, "active")
            self._mission_move_to(main_jump_plat, "middle", "하향 점프 테스트\n발판 중앙으로 이동하세요.")
            self._mission_action("DownJump", "아래 방향키 + 점프 (하향 점프)!", 
                                 repeat=self.mission_reps[mid], wait=2.0)
            self._update_mission_status(mid, "done")

            # 10. 급정거/방향전환
            mid = "M10"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "방향 전환(브레이킹) 테스트\n긴 발판 왼쪽 끝으로 이동하세요.")
            self._mission_action("Brake_Right_to_Left", "오른쪽으로 달리다가 급격히 왼쪽 키 입력!", 
                                 duration=2.5, wait=2.0, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            self._update_gui("🎉 모든 훈련 완료! 데이터가 저장되었습니다.")
            time.sleep(3)
            self._stop_training()

            # 11. 공중 역추진 (Air Brake)
            mid = "M11"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "공중 제어(역추진) 테스트\n긴 발판 왼쪽 끝으로 이동하세요.")
            # 우측 점프 후 공중에서 왼쪽 키 입력
            self._mission_action("Jump_Right_AirBrake", "우측 점프 후 공중에서 왼쪽 키(역추진)!", 
                                 duration=2.0, wait=1.5, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # 12. 공격 관성 (Attack Slide)
            mid = "M12"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "공격 관성 테스트\n긴 발판 왼쪽 끝으로 이동하세요.")
            # 우측 이동 중 공격 키 입력
            self._mission_action("Run_Attack_Right", "달리다가 공격(Attack) 키 입력!", 
                                 duration=2.5, wait=1.5, repeat=self.mission_reps[mid])
            self._update_mission_status(mid, "done")

            # --- 로프/사다리 데이터 확인 ---
            ropes = self.agent.map_processor.map_data.get("ropes", [])
            target_rope = ropes[0] if ropes else None

            if target_rope:
                # 로프 좌표 계산 (시각화 및 이동용)
                rx = target_rope['x']
                ry = target_rope['y']
                # 로프 근처 발판 찾기 (로프 X좌표를 포함하는 발판)
                rope_plat = next((p for p in platforms if p['x_start'] <= rx <= p['x_end'] and p['y'] > ry), None)
                
                if rope_plat:
                    # 13. 로프 매달리기
                    mid = "M13"
                    self._update_mission_status(mid, "active")
                    self._mission_move_to(rope_plat, "middle", "로프 테스트 준비\n로프 근처로 이동 중...")
                    
                    # 로프 위치로 정확히 이동하도록 가이드 (x_start, x_end를 로프 x좌표 근처로 설정)
                    self._update_gui(f"로프(x:{rx}) 아래로 이동해서 위쪽 방향키를 누르세요.", (rx-10, rx+10, rope_plat['y']))
                    time.sleep(3) # 유저가 로프를 탈 시간 부여
                    
                    self._mission_action("Rope_Climb_Stop", "로프에서 위/아래 이동 후 정지", 
                                         duration=3.0, wait=1.0, repeat=self.mission_reps[mid])
                    self._update_mission_status(mid, "done")

                    # 14. 로프 이탈 점프
                    mid = "M14"
                    self._update_mission_status(mid, "active")
                    self._mission_action("Rope_Jump_Away", "로프에 매달린 상태에서 점프!", 
                                         duration=1.5, wait=2.0, repeat=self.mission_reps[mid])
                    self._update_mission_status(mid, "done")
                else:
                    print("⚠️ 로프 아래에 밟을 수 있는 발판이 없어 로프 미션을 건너뜁니다.")
            else:
                print("⚠️ 맵 데이터에 'ropes'가 없어 로프 미션을 건너뜁니다.")

        except Exception as e:
            print("❌ 훈련 중 오류 발생:")
            traceback.print_exc()
            self._update_gui(f"오류 발생: {e}")

    def _visualizer_loop(self):
        if not self.is_running: return
        try:
            pos = self._get_player_pos()
            if pos:
                self.canvas.update_player(pos[0], pos[1])
        except Exception: pass
        self.root.after(100, self._visualizer_loop)

    def _mission_move_to(self, plat, position, msg):
        if self.agent:
            self.agent.is_recording = False
        
        padding = 30
        plat_w = plat['x_end'] - plat['x_start']
        
        if position == "middle":
            target_x_start = plat['x_start'] + plat_w // 3
            target_x_end = plat['x_end'] - plat_w // 3
        elif position == "left_edge":
            target_x_start = plat['x_start']
            target_x_end = plat['x_start'] + padding * 2
        elif position == "right_edge":
            target_x_start = plat['x_end'] - padding * 2
            target_x_end = plat['x_end']
        
        self._update_gui(msg, (target_x_start, target_x_end, plat['y']))

        while self.is_running:
            pos = self._get_player_pos()
            if pos:
                px, py = pos
                if (target_x_start <= px <= target_x_end) and abs(py - plat['y']) <= 15:
                    break
            time.sleep(0.5)
        
        self._update_gui("✅ 위치 도착! 준비하세요...", None)
        time.sleep(1.0)

    def _mission_action(self, scenario_name, msg, duration=0, repeat=1, wait=2.0):
        # repeat 인자가 0 이하로 들어오면 1회로 보정
        repeat = max(1, repeat)
        
        for i in range(repeat):
            if not self.is_running: return
            for c in range(3, 0, -1):
                self._update_gui(f"{msg}\n({i+1}/{repeat}) ⏳ {c}...", None)
                time.sleep(1)
            
            self._update_gui(f"🔥 GO! ({i+1}/{repeat})", None)
            
            if self.agent and self.agent.recorder:
                 mission_filename = f"Trainer_{scenario_name}_{i+1}"
                 self.agent.recorder.open(mission_filename)
                 self.agent.is_recording = True

            if duration > 0: time.sleep(duration)
            else: time.sleep(1.0)
                
            self._update_gui("🛑 멈추세요 (기록 중...)", None)
            
            self._wait_until_stopped(wait)

            if self.agent and self.agent.recorder:
                 self.agent.is_recording = False
                 self.agent.recorder.close()

    def _wait_until_stopped(self, timeout=2.0):
        start = time.time()
        last_pos = None
        stable_cnt = 0
        while time.time() - start < timeout:
            pos = self._get_player_pos()
            if pos and last_pos:
                if abs(pos[0] - last_pos[0]) < 2 and abs(pos[1] - last_pos[1]) < 2:
                    stable_cnt += 1
                else: stable_cnt = 0
            last_pos = pos
            if stable_cnt > 5: break
            time.sleep(0.1)

if __name__ == "__main__":
    real_logger = logger.logger if hasattr(logger, 'logger') else logger
    if hasattr(real_logger, 'handlers'):
        if not any(isinstance(h, logging.StreamHandler) for h in real_logger.handlers):
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            real_logger.addHandler(console)
            print("✅ Console logger attached.")

    root = tk.Tk()
    app = PhysicsTrainerVisualApp(root)
    root.mainloop()