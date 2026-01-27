# ui/tabs/map_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from modules.map_creator import MapCreator  # [신규] 분리된 로직 모듈 임포트

class MapTab:
    def __init__(self, notebook, agent, save_callback=None):
        self.agent = agent
        self.save_callback = save_callback  # 콜백 저장
        
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Map & AI Model")
        
        self.map_offset_x = 0
        self.map_offset_y = 0
        
        self._setup_ui()

    def _setup_ui(self):
        # 1. 맵 로드 (기존 코드 유지)
        map_frame = ttk.LabelFrame(self.frame, text="Map Data (.json)")
        map_frame.pack(fill="x", pady=5)
        self.lbl_map = ttk.Label(map_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map.pack(pady=2)
        ttk.Button(map_frame, text="📂 Load Map JSON", command=self.load_map_file).pack(fill="x", padx=5, pady=5)
        
        # 2. 오프셋 조정 (기존 코드 유지)
        offset_frame = ttk.LabelFrame(self.frame, text="Position Offset Correction")
        offset_frame.pack(fill="x", pady=5)
        self.lbl_offset = ttk.Label(offset_frame, text="Offset: (0, 0)", font=("Arial", 10, "bold"))
        self.lbl_offset.pack(pady=2)
        
        btn_pad = ttk.Frame(offset_frame)
        btn_pad.pack(pady=2)
        ttk.Button(btn_pad, text="▲", width=3, command=lambda: self.adjust_offset(0, -1)).grid(row=0, column=1)
        ttk.Button(btn_pad, text="◀", width=3, command=lambda: self.adjust_offset(-1, 0)).grid(row=1, column=0)
        ttk.Button(btn_pad, text="▼", width=3, command=lambda: self.adjust_offset(0, 1)).grid(row=1, column=1)
        ttk.Button(btn_pad, text="▶", width=3, command=lambda: self.adjust_offset(1, 0)).grid(row=1, column=2)
        ttk.Button(offset_frame, text="Reset", command=lambda: self.adjust_offset(0, 0, reset=True)).pack(pady=2)

        # 3. AI Models (LSTM + Physics)
        model_frame = ttk.LabelFrame(self.frame, text="AI Models")
        model_frame.pack(fill="x", pady=5)
        
        # [기존] LSTM
        self.lbl_lstm = ttk.Label(model_frame, text="LSTM: Not Loaded", foreground="gray")
        self.lbl_lstm.pack()
        ttk.Button(model_frame, text="🧠 Load LSTM", command=self.load_lstm).pack(fill="x", padx=5, pady=2)

        # [▼ 추가됨] Physics Engine
        ttk.Separator(model_frame, orient='horizontal').pack(fill='x', pady=5) # 구분선
        self.lbl_physics = ttk.Label(model_frame, text="Physics: Not Loaded", foreground="gray")
        self.lbl_physics.pack()
        ttk.Button(model_frame, text="⚛️ Load Physics", command=self.load_physics_model).pack(fill="x", padx=5, pady=2)

    # 4. [신규 UI] 맵 제작 도구 (Map Creator)
        # ==========================================
        self._setup_creator_ui()

    def _setup_creator_ui(self):
        """맵 제작 툴 UI 구성"""
        creator_frame = ttk.LabelFrame(self.frame, text="Map Creator Tool")
        creator_frame.pack(fill="x", pady=10)

        # 현재 좌표 모니터링
        self.lbl_current_pos = ttk.Label(creator_frame, text="Last Known Pos: (Wait...)", foreground="blue")
        self.lbl_current_pos.pack(pady=2)
        ttk.Button(creator_frame, text="🔄 Refresh Position Info", command=self.refresh_pos_info).pack(fill="x", padx=5, pady=2)

        # 시작점/종료점 표시 영역
        info_grid = ttk.Frame(creator_frame)
        info_grid.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(info_grid, text="Start Point:").grid(row=0, column=0, sticky="w")
        self.lbl_start_pos = ttk.Label(info_grid, text="Not Set", foreground="red")
        self.lbl_start_pos.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(info_grid, text="End Point:").grid(row=1, column=0, sticky="w")
        self.lbl_end_pos = ttk.Label(info_grid, text="Not Set", foreground="red")
        self.lbl_end_pos.grid(row=1, column=1, sticky="w", padx=5)

        # 조작 버튼
        btn_grid = ttk.Frame(creator_frame)
        btn_grid.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_grid, text="1. Set Start", command=self.on_set_start).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_grid, text="2. Set End", command=self.on_set_end).pack(side="left", expand=True, fill="x", padx=1)
        
        ttk.Button(creator_frame, text="3. ➕ Add Platform", command=self.on_add_platform).pack(fill="x", padx=5, pady=2)
        
        # 상태 표시
        self.lbl_platform_count = ttk.Label(creator_frame, text="Platforms: 0", font=("Arial", 9, "bold"))
        self.lbl_platform_count.pack(pady=2)

        # 저장 버튼
        ttk.Separator(creator_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(creator_frame, text="💾 Save New Map JSON", command=self.on_save_map).pack(fill="x", padx=5, pady=5)

    # --- Event Handlers (UI Logic) ---

    def refresh_pos_info(self):
        """현재 좌표 UI 갱신"""
        pos = self.map_creator.get_current_pos()
        self.lbl_current_pos.config(text=f"Last Known Pos: {pos}")

    def on_set_start(self):
        """시작점 설정 버튼 핸들러"""
        success, pos = self.map_creator.set_start_point()
        self.refresh_pos_info()
        
        if success:
            self.lbl_start_pos.config(text=f"{pos}", foreground="green")
        else:
            messagebox.showwarning("Warning", "플레이어 위치를 인식할 수 없습니다.\n미니맵에 노란 점이 보이는지 확인하세요.")

    def on_set_end(self):
        """종료점 설정 버튼 핸들러"""
        success, pos = self.map_creator.set_end_point()
        self.refresh_pos_info()
        
        if success:
            self.lbl_end_pos.config(text=f"{pos}", foreground="green")
        else:
            messagebox.showwarning("Warning", "플레이어 위치를 인식할 수 없습니다.\n미니맵에 노란 점이 보이는지 확인하세요.")

    def on_add_platform(self):
        """발판 추가 버튼 핸들러"""
        if not self.map_creator.is_ready_to_add():
            messagebox.showerror("Error", "시작점과 종료점을 모두 설정해야 합니다.")
            return

        success, new_plat = self.map_creator.add_platform()
        if success:
            # UI 초기화
            self.lbl_start_pos.config(text="Not Set", foreground="red")
            self.lbl_end_pos.config(text="Not Set", foreground="red")
            
            count = self.map_creator.get_platform_count()
            self.lbl_platform_count.config(text=f"Platforms: {count}")
            print(f"[MapTab] Platform Added: {new_plat}")

    def on_save_map(self):
        """저장 버튼 핸들러"""
        if self.map_creator.get_platform_count() == 0:
            messagebox.showwarning("Warning", "저장할 발판 데이터가 없습니다.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialfile="new_map.json"
        )
        
        if file_path:
            success, msg = self.map_creator.save_map_to_json(file_path)
            if success:
                messagebox.showinfo("Success", f"맵 파일이 저장되었습니다.\n{os.path.basename(file_path)}")
                if messagebox.askyesno("Reset", "저장 후 작업 내역을 초기화하시겠습니까?"):
                    self.map_creator.clear_data()
                    self.lbl_platform_count.config(text="Platforms: 0")
            else:
                messagebox.showerror("Error", f"저장 실패: {msg}")

    def update_file_label(self, file_type, path):
        filename = os.path.basename(path)
        if file_type == "map":
            self.lbl_map.config(text=filename, foreground="green")
        elif file_type == "model":
            self.lbl_lstm.config(text=filename, foreground="blue")
        elif file_type == "physics":
            self.lbl_physics.config(text=filename, foreground="blue")

    def load_map_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            if self.agent.load_map(path):
                self.lbl_map.config(text=os.path.basename(path), foreground="green")
                # [신규] 설정 저장 호출
                if self.save_callback: 
                    self.save_callback(map_path=path)
            else:
                messagebox.showerror("에러", "맵 로드 실패")

    def load_lstm(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            if self.agent.model_loader.load_model(path):
                self.lbl_lstm.config(text=os.path.basename(path), foreground="blue")
                # [신규] 설정 저장 호출
                if self.save_callback: 
                    self.save_callback(model_path=path)
            else:
                messagebox.showerror("에러", "LSTM/GRU 모델 로드 실패")

    def load_physics_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            if hasattr(self.agent, 'physics_engine') and self.agent.physics_engine:
                if self.agent.physics_engine.load_model(path):
                    if hasattr(self, 'lbl_physics'):
                        self.lbl_physics.config(text=os.path.basename(path), foreground="blue")
                    # [신규] 설정 저장 호출
                    if self.save_callback: 
                        self.save_callback(physics_path=path)
                    return
            
            messagebox.showerror("에러", "물리 엔진 로드 실패\n(BotAgent 초기화를 확인하세요)")

    def adjust_offset(self, dx, dy, reset=False):
        # (기존 코드 유지)
        if reset:
            self.map_offset_x = 0
            self.map_offset_y = 0
        else:
            self.map_offset_x += dx
            self.map_offset_y += dy
            
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")

        if self.agent:
            self.agent.set_map_offset(self.map_offset_x, self.map_offset_y)

    def update_info(self, map_path=None, lstm_path=None, rf_path=None):
        """외부에서 로드된 경로 정보를 받아 UI 라벨을 갱신합니다."""
        if map_path and os.path.exists(map_path):
            # 맵 이름만 추출하여 표시 (예: "map_1.json")
            name = os.path.basename(map_path)
            # self.lbl_map_name 등의 변수명은 사용하시는 코드에 맞게 확인 필요
            # 만약 라벨 변수가 self.lbl_current_map 이라면:
            if hasattr(self, 'lbl_map_name'):
                self.lbl_map_name.config(text=f"현재 맵: {name}")
            elif hasattr(self, 'lbl_cur_map'): # 변수명이 다를 경우 대비
                self.lbl_cur_map.config(text=f"현재 맵: {name}")

        if lstm_path and os.path.exists(lstm_path):
            name = os.path.basename(lstm_path)
            if hasattr(self, 'lbl_lstm_name'):
                self.lbl_lstm_name.config(text=f"AI 모델: {name}")

        if rf_path and os.path.exists(rf_path):
            name = os.path.basename(rf_path)
            if hasattr(self, 'lbl_rf_name'):
                self.lbl_rf_name.config(text=f"물리 모델: {name}")
                
        print(f"UI 업데이트 완료: {map_path}, {lstm_path}")