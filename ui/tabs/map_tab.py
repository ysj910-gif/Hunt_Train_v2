# ui/tabs/map_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog # [수정] simpledialog 추가
import os
from modules.map_creator import MapCreator  # [신규] 분리된 로직 모듈 임포트

class MapTab:
    def __init__(self, notebook, agent, save_callback=None):
        self.agent = agent
        self.save_callback = save_callback  # 콜백 저장

        self.map_creator = MapCreator(self.agent)
        
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Map Tool")
        
        self.map_offset_x = 0
        self.map_offset_y = 0
        
        self._setup_ui()

        self._update_loop()

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

        # 3. 맵 제작 도구 (Map Creator)
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
        # ========================================================
        # 1. Set Start / 2. Set End 버튼 영역
        # ========================================================
        btn_grid = ttk.Frame(creator_frame)
        btn_grid.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_grid, text="1. Set Start", command=self.on_set_start).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_grid, text="2. Set End", command=self.on_set_end).pack(side="left", expand=True, fill="x", padx=1)

        
        # 3. [수정] 객체 추가 버튼 영역 (그리드로 변경하여 배치)
        add_frame = ttk.LabelFrame(creator_frame, text="Add Objects")
        add_frame.pack(fill="x", padx=5, pady=5)

        # [신규] 맨 아래 발판 체크박스
        self.var_is_bottom = tk.BooleanVar(value=False)
        ttk.Checkbutton(add_frame, text="맨 아래 발판 (⬇️점프 불가)", variable=self.var_is_bottom).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Row 0: 기본 구조물
        ttk.Button(add_frame, text="🧱 Platform", command=self.on_add_platform).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(add_frame, text="🪢 Rope", command=self.on_add_rope).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        # Row 1: 포탈류
        ttk.Button(add_frame, text="🌀 Local Portal", command=self.on_add_portal).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(add_frame, text="🚪 Map Portal (Next Map)", command=self.on_add_map_portal).grid(row=1, column=1, sticky="ew", padx=2, pady=2) # [신규]
        
        # 그리드 비율 조정
        add_frame.columnconfigure(0, weight=1)
        add_frame.columnconfigure(1, weight=1)

        # (4) [신규] 스폰 매니저
        spawn_frame = ttk.LabelFrame(creator_frame, text="Spawn Manager (Auto Distribute)")
        spawn_frame.pack(fill="x", padx=5, pady=5)
        
        input_frame = ttk.Frame(spawn_frame)
        input_frame.pack(fill="x", pady=2)
        ttk.Label(input_frame, text="Total Mob Count:").pack(side="left", padx=5)
        
        self.ent_spawn_count = ttk.Entry(input_frame, width=5)
        self.ent_spawn_count.insert(0, "28") # Default
        self.ent_spawn_count.pack(side="left")
        
        ttk.Button(input_frame, text="⚡ Generate", command=self.on_generate_spawns).pack(side="left", padx=5)
        
        ttk.Button(spawn_frame, text="❌ Add No-Spawn Zone (Here)", 
                   command=self.on_add_no_spawn_zone).pack(fill="x", padx=5, pady=2)

        # 4. [신규] 실행 취소 버튼
        ttk.Button(creator_frame, text="↩️ Undo Last Action", command=self.on_undo).pack(fill="x", padx=5, pady=2)

        # 5. [수정] 상태 표시 (종합 정보)
        self.lbl_status = ttk.Label(creator_frame, text="Ready", font=("Arial", 9))
        self.lbl_status.pack(pady=2)
        
        # 저장 버튼
        ttk.Separator(creator_frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(creator_frame, text="💾 Save New Map JSON", command=self.on_save_map).pack(fill="x", padx=5, pady=5)

    def _update_loop(self):
        """실시간 좌표 갱신 (100ms 간격)"""
        # 탭(프레임)이 살아있을 때만 동작
        if self.frame.winfo_exists():
            pos = self.map_creator.get_current_pos()
            self.lbl_current_pos.config(text=f"Last Known Pos: {pos}")
            # 100ms 뒤에 다시 자기 자신 호출 (재귀적 루프)
            self.frame.after(100, self._update_loop)

    # --- Event Handlers (UI Logic) ---

    def refresh_pos_info(self):
        """현재 좌표 UI 갱신 (유지)"""
        pos = self.map_creator.get_current_pos()
        self.lbl_current_pos.config(text=f"Last Known Pos: {pos}")

    def on_set_start(self):
        """시작점 설정 버튼 핸들러 (유지)"""
        success, pos = self.map_creator.set_start_point()
        self.refresh_pos_info()
        
        if success:
            self.lbl_start_pos.config(text=f"{pos}", foreground="green")
        else:
            messagebox.showwarning("Warning", "플레이어 위치를 인식할 수 없습니다.\n미니맵에 노란 점이 보이는지 확인하세요.")

    def on_set_end(self):
        """종료점 설정 버튼 핸들러 (유지)"""
        success, pos = self.map_creator.set_end_point()
        self.refresh_pos_info()
        
        if success:
            self.lbl_end_pos.config(text=f"{pos}", foreground="green")
        else:
            messagebox.showwarning("Warning", "플레이어 위치를 인식할 수 없습니다.\n미니맵에 노란 점이 보이는지 확인하세요.")

    # [신규] 공통 UI 업데이트 헬퍼 메서드
    def _update_status_ui(self):
        """작업 후 UI 상태(라벨 등)를 일괄 갱신합니다."""
        # 1. 시작/종료점 라벨 초기화
        self.lbl_start_pos.config(text="Not Set", foreground="red")
        self.lbl_end_pos.config(text="Not Set", foreground="red")
        
        # 2. 종합 상태 표시 (MapCreator.get_summary 활용)
        if hasattr(self, 'lbl_status'): # lbl_status가 없는 경우 lbl_platform_count 사용
            summary = self.map_creator.get_summary()
            self.lbl_status.config(text=summary, foreground="blue")
        else:
            # 기존 라벨 호환성
            count = self.map_creator.get_platform_count()
            self.lbl_platform_count.config(text=f"Objects: {count}")

    def on_add_platform(self):
        """발판 추가 버튼 핸들러 (맨 아래 발판 옵션 적용)"""
        # 체크박스 값 가져오기 (UI에 self.var_is_bottom이 정의되어 있어야 함)
        is_bottom = self.var_is_bottom.get()
        
        success, res = self.map_creator.add_platform(is_bottom=is_bottom)
        if success:
            self._update_status_ui()
            print(f"[MapTab] Platform Added: {res}")
        else:
            messagebox.showwarning("Error", res)

    def on_generate_spawns(self, silent=False):
        """스폰 포인트 생성 및 재분배"""
        try:
            count = int(self.ent_spawn_count.get())
            success, msg = self.map_creator.generate_spawns(count)
            if success:
                self._update_status_ui()
                # silent=True일 경우 메시지 창 생략 (자동 재배치용)
                if not silent:
                    messagebox.showinfo("Spawns", msg)
            else:
                if not silent:
                    messagebox.showerror("Error", msg)
        except ValueError:
            messagebox.showerror("Error", "몬스터 수에 숫자를 입력하세요.")

    def on_add_no_spawn_zone(self):
        """[신규] 스폰 제외 구역 추가 (현재 위치 기준)"""
        # 1. 금지 구역 추가 (좌우 50px)
        success, msg = self.map_creator.add_no_spawn_zone(radius=50)
        
        if success:
            # 2. 성공 시 즉시 몬스터 재배치 (조용히 실행)
            self.on_generate_spawns(silent=True)
            
            # 3. 결과 알림
            messagebox.showinfo("Zone Added", f"{msg}\n\n해당 구역을 피해 몬스터가 재배치되었습니다.")
        else:
            messagebox.showwarning("Warning", msg)

    def on_add_portal(self):
        """[신규] 포탈 추가 버튼 핸들러"""
        if not self.map_creator.is_ready_to_add():
            messagebox.showerror("Error", "시작점과 종료점을 모두 설정해야 합니다.")
            return

        success, res = self.map_creator.add_portal()
        if success:
            self._update_status_ui()
            print(f"[MapTab] Portal Added: {res}")
        else:
            messagebox.showwarning("Error", res)

    def on_add_rope(self):
        """[신규] 밧줄 추가 버튼 핸들러"""
        if not self.map_creator.is_ready_to_add():
            messagebox.showerror("Error", "시작점과 종료점을 모두 설정해야 합니다.")
            return

        success, res = self.map_creator.add_rope()
        if success:
            self._update_status_ui()
            print(f"[MapTab] Rope Added: {res}")
        else:
            messagebox.showwarning("Error", res)

    def on_add_map_portal(self):
        """[신규] 맵 이동 포탈 추가 핸들러"""
        # 1. 위치 설정 확인 (시작점만 있으면 됨)
        if self.map_creator.temp_start_pos is None:
            messagebox.showwarning("Warning", "포탈 위치(Start Point)를 먼저 설정해주세요.")
            return

        # 2. 이동할 맵 이름 입력
        target_name = simpledialog.askstring("Map Portal", "이동할 맵 이름을 입력하세요:\n(예: El Nath, Henesys)")
        
        if target_name:
            success, res = self.map_creator.add_map_portal(target_name)
            if success:
                self._update_status_ui()
                print(f"[MapTab] Map Portal Added: {res}")
            else:
                messagebox.showerror("Error", res)

    def on_undo(self):
        """실행 취소 (금지 구역 취소 시 스폰 복구 포함)"""
        success, msg = self.map_creator.undo_last_action()
        
        if success:
            self._update_status_ui()
            
            if "no_spawn" in msg:
                self.on_generate_spawns(silent=True)
                msg += "\n(스폰 포인트가 빈 자리에 다시 채워졌습니다.)"
                
            messagebox.showinfo("Undo", msg)
        else:
            messagebox.showwarning("Undo", msg)

    def on_save_map(self):
        """저장 버튼 핸들러 (수정됨)"""
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
                    self._update_status_ui() # UI 초기화
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
            name = os.path.basename(map_path)
            # [수정] _setup_ui에서 생성한 변수명(self.lbl_map) 사용
            if hasattr(self, 'lbl_map'):
                self.lbl_map.config(text=f"현재 맵: {name}", foreground="green")
        
        # (LSTM 부분도 동일하게 self.lbl_lstm으로 통일 권장)
        if lstm_path and os.path.exists(lstm_path):
            name = os.path.basename(lstm_path)
            if hasattr(self, 'lbl_lstm'):
                self.lbl_lstm.config(text=f"LSTM: {name}", foreground="blue")

        # (Physics 부분도 self.lbl_physics로 통일 권장)
        if rf_path and os.path.exists(rf_path):
            name = os.path.basename(rf_path)
            if hasattr(self, 'lbl_physics'):
                self.lbl_physics.config(text=f"Physics: {name}", foreground="blue")
                
        print(f"UI 업데이트 완료: {map_path}, {lstm_path}")