#physics\physics_trainer

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import time
import sys
import os
import json
import traceback
import logging
import cv2  # [추가] OpenCV

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [필수 모듈 임포트]
try:
    from core.bot_agent import BotAgent
    from utils.logger import logger, trace_logic  # [추가] trace_logic 임포트
    from modules.vision_system import VisionSystem
    from modules.scanner import GameScanner
    from core.action_handler import ActionHandler
    from engine.map_processor import MapProcessor
    from engine.path_finder import PathFinder
    from engine.physics_engine import PhysicsEngine
    from core.decision_maker import DecisionMaker  
    from core.data_recorder import DataRecorder
    from ui.components.roi_selector import ROISelector # [추가] ROI 선택기 임포트
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
        self.root.geometry("1150x850") # [수정] 높이 약간 증가
        
        # [수정] Vision 및 Scanner 미리 초기화 (ROI 설정을 위해)
        self.vision = VisionSystem()
        self.scanner = GameScanner()
        
        self.agent = None
        self.is_running = False
        self.is_paused = False

        self.current_loop_idx = 0 
        self.total_repeat_count = 0

        self.map_path = tk.StringVar()
        self.instruction_text = tk.StringVar(value="맵 파일을 로드해주세요.")
        
        # 미션 목록 정의
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
            ("M11", "공중 역추진 (Air Brake)", 15),
            ("M12", "공격 관성 (이동 중 공격)", 15),
            ("M13", "로프 매달리기/이동", 10),
            ("M14", "로프 이탈 점프", 10),
            ("M15", "상향 점프 (제자리)", 30),   # [추가] 제자리 상향 점프
            ("M16", "상향 점프 (이동)", 30)
        ]
        
        self.mission_reps = {mid: default_reps for mid, _, default_reps in self.missions_data}

        self._setup_ui()

    def _setup_ui(self):
        # 1. 상단 설정
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")
        ttk.Label(top_frame, text="맵 파일:").pack(side="left")
        ttk.Entry(top_frame, textvariable=self.map_path, width=40).pack(side="left", padx=5)
        ttk.Button(top_frame, text="📂 열기", command=self._browse_map).pack(side="left")
        
        # [추가] ROI 설정 버튼
        ttk.Separator(top_frame, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(top_frame, text="🗺️ 미니맵 설정", command=self._set_minimap_roi).pack(side="left", padx=5)
        
        # [추가] 창 위치 갱신 버튼
        ttk.Button(top_frame, text="🔄 창 위치 갱신", command=self._refresh_window_position).pack(side="left", padx=5)
        
        ttk.Button(top_frame, text="🛑 중지", command=self._stop_training).pack(side="right")

        # 2. 중단 (맵 + 체크리스트)
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        map_frame = ttk.LabelFrame(mid_frame, text="📍 실시간 맵 & 위치")
        map_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.canvas = MapVisualizer(map_frame, bg="#f5f5f5")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        list_frame = ttk.LabelFrame(mid_frame, text="📋 To-Do Check List", width=320)
        list_frame.pack(side="right", fill="y")
        
        cols = ("status", "name", "reps")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=20)
        
        self.tree.heading("status", text="상태")
        self.tree.heading("name", text="미션 내용")
        self.tree.heading("reps", text="반복")
        
        self.tree.column("status", width=50, anchor="center")
        self.tree.column("name", width=180)
        self.tree.column("reps", width=50, anchor="center")
        
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)
        self.tree.bind("<Double-1>", self._on_tree_double_click)
        
        for mid, name, reps in self.missions_data:
            self.tree.insert("", "end", iid=mid, values=("⬜", name, f"{reps}회"))

        # 3. 하단
        bottom_frame = ttk.LabelFrame(self.root, text="📢 코치 지시사항", padding=10)
        bottom_frame.pack(fill="x", padx=10, pady=10)
        
        self.lbl_instruction = ttk.Label(bottom_frame, textvariable=self.instruction_text, 
                                         font=("Helvetica", 16, "bold"), foreground="blue", anchor="center")
        self.lbl_instruction.pack(fill="x", pady=(0, 10))
        
        # [추가] 컨트롤 패널
        ctrl_frame = ttk.Frame(bottom_frame)
        ctrl_frame.pack(fill="x")
        
        # [수정] 버튼 3개 배치 (시작 / 일시정지 / 이전삭제)
        self.btn_start = ttk.Button(ctrl_frame, text="🚀 훈련 시작", command=self._start_training, state="disabled")
        self.btn_start.pack(side="left", fill="x", expand=True, ipady=5, padx=2)
        
        self.btn_pause = ttk.Button(ctrl_frame, text="⏸️ 일시정지", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, ipady=5, padx=2)
        
        self.btn_undo = ttk.Button(ctrl_frame, text="↩️ 최근 데이터 삭제", command=self._undo_last_data, state="disabled")
        self.btn_undo.pack(side="left", fill="x", expand=True, ipady=5, padx=2)
        
        # [추가] 실시간 화면 보기 체크박스
        self.show_vision_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="📸 실시간 인식(Vision) 확인", variable=self.show_vision_var).pack(side="right", padx=10)

    def _refresh_window_position(self):
        if self.vision.find_window():
            # 창을 다시 찾으면 캡처 영역(capture_area)이 갱신됨
            rect = self.vision.capture_area
            messagebox.showinfo("갱신 완료", f"메이플스토리 창을 다시 찾았습니다.\n위치: {rect}")
            # 필요하다면 ROI 재설정 알림을 줄 수도 있음
        else:
            messagebox.showerror("실패", "메이플스토리 창을 찾을 수 없습니다.\n게임을 실행했는지 확인해주세요.")

    # [추가] 미니맵 ROI 설정 메서드
    def _set_minimap_roi(self):
        # Vision System이 창을 못 찾으면 먼저 찾게 함
        if not self.vision.find_window():
            messagebox.showwarning("경고", "메이플스토리 창을 찾을 수 없습니다.\n게임을 실행해주세요.")
            return

        # ROISelector가 'agent.vision' 구조를 원하므로 Proxy 객체 생성
        class AgentProxy:
            def __init__(self, vision):
                self.vision = vision
        
        # 팝업 실행
        ROISelector(self.root, AgentProxy(self.vision), "minimap")
        logger.info("미니맵 설정 팝업 열림")

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

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶️ 다시 시작 (재개)")
            self.instruction_text.set("⏸️ 훈련이 일시정지 되었습니다.")
        else:
            self.btn_pause.config(text="⏸️ 일시정지")
            self.instruction_text.set("훈련을 재개합니다...")

    # [추가] 최근 데이터 삭제 메서드
    def _undo_last_data(self):
        if not self.agent or not self.agent.recorder:
            messagebox.showwarning("경고", "레코더가 초기화되지 않았습니다.")
            return
            
        last_file = self.agent.recorder.last_filepath
        
        if last_file and os.path.exists(last_file):
            confirm = messagebox.askyesno("삭제/되돌리기 확인", 
                f"가장 최근 기록을 삭제하고\n이번 회차({self.current_loop_idx}회)를 다시 수행하시겠습니까?\n\n파일: {os.path.basename(last_file)}")
            
            if confirm:
                try:
                    # 1. 파일 삭제
                    if self.agent.recorder.file and not self.agent.recorder.file.closed:
                        self.agent.recorder.close()
                    os.remove(last_file)
                    print(f"🗑️ 파일 삭제 완료: {last_file}")
                    
                    # 2. [핵심] 루프 카운트 되돌리기 (0보다 클 때만)
                    if self.current_loop_idx > 0:
                        self.current_loop_idx -= 1
                        self._update_gui(f"↩️ {self.current_loop_idx + 1}회차부터 다시 시작합니다...")
                    else:
                         messagebox.showinfo("알림", "현재 미션의 첫 번째 데이터 이전입니다.\n파일은 삭제되었으나 미션 단계는 유지됩니다.")

                except Exception as e:
                    messagebox.showerror("오류", f"작업 실패: {e}")
        else:
            messagebox.showinfo("알림", "삭제할 최근 기록 파일이 없습니다.")

    def _start_training(self):
        self.is_running = True
        self.is_paused = False  # 초기화
        
        # [수정] 시작 버튼은 끄고, 제어 버튼들은 켜기
        self.btn_start.config(state="disabled")
        
        # 아래 두 줄이 없어서 버튼이 계속 비활성화 상태였던 것입니다.
        self.btn_pause.config(state="normal", text="⏸️ 일시정지")
        self.btn_undo.config(state="normal")
        
        t = threading.Thread(target=self._training_routine)
        t.daemon = True
        t.start()

    def _stop_training(self):
        self.is_running = False
        self.is_paused = False
        if self.agent:
            self.agent.stop()
        self.instruction_text.set("훈련이 중지되었습니다.")
        
        # [수정] 버튼 상태 원상복구
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸️ 일시정지")
        self.btn_undo.config(state="disabled") # 훈련 종료 후 삭제 막으려면 disabled, 아니면 normal

    def _update_gui(self, text, target_area=None):
        self.instruction_text.set(text)
        if target_area:
            self.canvas.set_target(*target_area)
        else:
            if self.canvas.target_id:
                self.canvas.delete(self.canvas.target_id)
                self.canvas.target_id = None

    @trace_logic
    def _get_player_pos(self):
        # 1. Scanner가 직접 가지고 있는 위치 우선 (실시간성 높음)
        if self.scanner and self.scanner.last_player_pos != (0,0):
             return self.scanner.last_player_pos
             
        # 2. Agent를 통한 데이터 접근
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
            
            # [수정] 이미 생성된 self.vision, self.scanner 사용
            action_handler = ActionHandler() 
            map_processor = MapProcessor()
            physics_engine = PhysicsEngine()
            
            if os.path.exists("physics_hybrid_model.pth"):
                physics_engine.load_model("physics_hybrid_model.pth")
            
            path_finder = PathFinder(map_processor, physics_engine)
            recorder = DataRecorder(map_processor, "Session_Log")  

            self.agent = BotAgent(
                vision=self.vision,     # [연결]
                scanner=self.scanner,   # [연결]
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

            # 시각화 루프 시작 (훈련 시작 시점에 확실히 트리거)
            self.root.after(100, self._visualizer_loop)

            # ... (미션 수행 로직은 동일하므로 생략하거나 유지) ...
            platforms = self.agent.map_processor.platforms
            if not platforms: raise ValueError("No platforms found.")
            
            # --- (이하 미션 코드는 원본 유지) ---
            run_plat = max(platforms, key=lambda p: p['x_end'] - p['x_start'])
            jump_plats = sorted(platforms, key=lambda p: p['y'])
            main_jump_plat = jump_plats[0] if jump_plats else run_plat
            
            self._update_gui("⚠️ 훈련 세션이 시작되었습니다.")
            time.sleep(2)
            
            # 첫번째 미션
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
            ropes = []
            if self.canvas.map_data:
                ropes = self.canvas.map_data.get("ropes", [])
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

            mid = "M15"
            self._update_mission_status(mid, "active")
            # 넓은 발판(run_plat) 중앙으로 이동
            self._mission_move_to(run_plat, "middle", "상향 점프 (제자리) 테스트\n발판 중앙으로 이동하세요.")
            # Action: 제자리에서 윗점프
            self._mission_action("UpJump_Stationary", "윗방향키 + 점프 (상향 점프)!", 
                                 repeat=self.mission_reps[mid], wait=2.5)
            self._update_mission_status(mid, "done")

            # [추가] 16. 상향 점프 (이동)
            # 좌우 대칭이므로 한 방향(오른쪽)만 수행
            mid = "M16"
            self._update_mission_status(mid, "active")
            self._mission_move_to(run_plat, "left_edge", "상향 점프 (이동) 테스트\n발판 왼쪽 끝으로 이동하세요.")
            # Action: 오른쪽으로 달리면서 윗점프
            self._mission_action("UpJump_Move_Right", "오른쪽으로 달리면서 윗방향키 + 점프!", 
                                 duration=2.0, repeat=self.mission_reps[mid], wait=2.5)
            self._update_mission_status(mid, "done")

        except Exception as e:
            print("❌ 훈련 중 오류 발생:")
            traceback.print_exc()
            self._update_gui(f"오류 발생: {e}")

    def _visualizer_loop(self):
        if not self.is_running: return
        try:
            # 여기서는 float 좌표를 그대로 받아옴
            pos = self._get_player_pos()
            
            # [수정 1] 캔버스 업데이트 시 int 변환 불필요 (Tkinter는 float도 처리 가능하지만 안전하게 int 권장)
            if pos:
                self.canvas.update_player(pos[0], pos[1])
        except Exception: pass

        if self.show_vision_var.get():
            try:
                if not self.vision.window_found:
                    self.vision.find_window()
                
                frame = self.vision.capture()
                if frame is not None:
                    if self.vision.minimap_roi:
                        self.scanner.set_rois(self.vision.minimap_roi, self.vision.kill_roi)
                    
                    cx, cy = self.scanner.find_player(frame)
                    
                    # 그리기 (OpenCV)
                    if self.vision.minimap_roi:
                        mx, my, mw, mh = self.vision.minimap_roi
                        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
                        
                        if cx > 0 and cy > 0:
                            # [수정 2] 화면에 그릴 때만 int()로 변환하여 소수점 버림
                            # (데이터 기록이나 물리 엔진에는 소수점 그대로 전달됨)
                            screen_x = int(mx + cx)
                            screen_y = int(my + cy)
                            
                            cv2.circle(frame, (screen_x, screen_y), 5, (0, 0, 255), -1)
                            # 텍스트에는 소수점 둘째 자리까지 표시
                            cv2.putText(frame, f"Pos: {cx:.2f},{cy:.2f}", (screen_x+10, screen_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.imshow("Bot Vision Debug", frame)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Vision Error: {e}")
        else:
            # 체크박스 끄면 창 닫기
            try:
                if cv2.getWindowProperty("Bot Vision Debug", 0) >= 0:
                    cv2.destroyWindow("Bot Vision Debug")
            except: pass

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
        # 1. 상태 초기화
        self.current_loop_idx = 0
        self.total_repeat_count = max(1, repeat)
        
        # 2. While 루프로 변경 (Undo 기능을 위해)
        while self.current_loop_idx < self.total_repeat_count:
            i = self.current_loop_idx # 현재 회차 (0부터 시작)

            # ----------------------------------
            # (A) 훈련 중지/일시정지 체크 구역
            # ----------------------------------
            if not self.is_running: return
            while self.is_paused:
                if not self.is_running: return
                time.sleep(0.5)

            # ----------------------------------
            # (B) 카운트다운
            # ----------------------------------
            for c in range(3, 0, -1):
                # 카운트다운 중에도 Index가 바뀌거나(Undo) 일시정지될 수 있음
                while self.is_paused: 
                    time.sleep(0.5)
                if not self.is_running: return
                
                # GUI 업데이트 (현재 회차 표시)
                # i+1을 표시하지만, 사용자가 Undo를 누르면 self.current_loop_idx가 줄어들어
                # 다음 루프 때 숫자가 줄어든 상태로 다시 출력됨.
                current_display_idx = self.current_loop_idx + 1
                self._update_gui(f"{msg}\n({current_display_idx}/{self.total_repeat_count}) ⏳ {c}...", None)
                time.sleep(1)
            
            # ----------------------------------
            # (C) 액션 실행 (기록 시작)
            # ----------------------------------
            current_display_idx = self.current_loop_idx + 1
            self._update_gui(f"🔥 GO! ({current_display_idx}/{self.total_repeat_count})", None)
            
            if self.agent and self.agent.recorder:
                 # 파일명에 회차 번호 포함
                 mission_filename = f"Trainer_{scenario_name}_{current_display_idx}"
                 self.agent.recorder.open(mission_filename)
                 self.agent.is_recording = True

            if duration > 0: time.sleep(duration)
            else: time.sleep(1.0)
                
            self._update_gui("🛑 멈추세요 (기록 중...)", None)
            
            # ----------------------------------
            # (D) 안정화 대기 및 기록 종료
            # ----------------------------------
            self._wait_until_stopped(wait)

            if self.agent and self.agent.recorder:
                 self.agent.is_recording = False
                 self.agent.recorder.close()
            
            # ----------------------------------
            # (E) 루프 인덱스 증가
            # ----------------------------------
            # 여기서 증가시키므로, 만약 위 과정 후 Undo를 누르면 
            # self.current_loop_idx가 다시 1 줄어들어 while 조건문 안에서 다시 실행됨
            self.current_loop_idx += 1
            
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