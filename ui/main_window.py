# ui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

# 모듈 임포트
from modules.job_manager import JobManager
from ui.tabs.skill_tab import SkillTab
from ui.tabs.map_tab import MapTab
from ui.tabs.simulation_tab import SimulationTab
from ui.tabs.engine_tab import EngineTab  # [추가]
from ui.components.status_panel import StatusPanel  # [신규 추가]
from ui.components.roi_selector import ROISelector
from ui.components.visualizer import Visualizer
from ui.components.viewport_manager import ViewportManager
from ui.components.simulation_mode import SimulationMode


class MainWindow:
    def __init__(self, root, agent):
        self.root = root
        self.agent = agent
        
        self.root.title("MapleHunter v2.0 Modular UI")
        self.root.geometry("1300x900") # 너비 약간 확장
        
        self.job_mgr = JobManager()
        self.config_path = "config.json"
        
        self.cur_map_path = ""
        self.cur_lstm_path = ""
        self.cur_rf_path = ""

        self.skill_tab = None
        self.map_tab = None
        self.simulation_tab = None  # [추가] 변수 초기화
        self.engine_tab = None
        self.status_panel = None # [신규]
        self.sim_mode = None # 시뮬레이션 모드 객체
        self.is_simulating = False

        self.viewport = ViewportManager() 
        self.last_mouse_pos = (0, 0)
        self.view_scale = 1.0

        self.sim_mode = None
        self.is_simulating = False
        
        self.setup_ui()
        self.load_settings()
        self.update_ui_loop()
        

    def setup_ui(self):
        # 1. 메인 좌우 분할
        self.main_split = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_split.pack(fill="both", expand=True)

        self.frame_left_container = ttk.Frame(self.main_split)
        self.frame_right = ttk.Frame(self.main_split, width=420)
        
        self.main_split.add(self.frame_left_container, weight=3) # 좌측 비중 큼
        self.main_split.add(self.frame_right, weight=1)

        # 2. [Left] 상하 분할 (게임화면 / 상태창) - ★ 핵심 수정 부분
        self.left_split = ttk.PanedWindow(self.frame_left_container, orient=tk.VERTICAL)
        self.left_split.pack(fill="both", expand=True)
        
        # 2-1. 상단: 게임 화면
        self.canvas_frame = ttk.Frame(self.left_split)
        self.left_split.add(self.canvas_frame, weight=3) # 화면 영역 크게

        # [신규] 줌 컨트롤 바 추가 (캔버스 바로 위에 배치)
        zoom_frame = ttk.Frame(self.canvas_frame)
        zoom_frame.pack(side="top", fill="x", padx=5, pady=2)
        
        ttk.Label(zoom_frame, text="View Zoom:").pack(side="left")
        
        # 줌 아웃 버튼
        btn_minus = ttk.Button(zoom_frame, text="🔍-", width=3, command=lambda: self.change_zoom(-0.2))
        btn_minus.pack(side="left", padx=2)
        
        # 현재 배율 표시 라벨
        self.lbl_zoom = ttk.Label(zoom_frame, text="100%", width=6, anchor="center")
        self.lbl_zoom.pack(side="left", padx=2)
        
        # 줌 인 버튼
        btn_plus = ttk.Button(zoom_frame, text="🔍+", width=3, command=lambda: self.change_zoom(0.2))
        btn_plus.pack(side="left", padx=2)
        
        # 리셋 버튼
        btn_reset = ttk.Button(zoom_frame, text="Reset", width=5, command=lambda: self.change_zoom(0, reset=True))
        btn_reset.pack(side="left", padx=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # 2-2. 하단: 상태 및 로그 패널
        self.status_frame = ttk.Frame(self.left_split)
        self.left_split.add(self.status_frame, weight=1) # 정보 영역 작게
        
        self.status_panel = StatusPanel(self.status_frame) # 모듈 연결

        # 3. [Right] 우측 탭 구성
        self.tabs = ttk.Notebook(self.frame_right)
        self.tabs.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.skill_tab = SkillTab(self.tabs, self.agent, self.job_mgr, self.save_settings)
        self.skill_tab.on_job_change_callback = self.on_job_change_handler
        
        # [수정] MapTab에도 save_settings 콜백 전달
        self.map_tab = MapTab(self.tabs, self.agent, self.save_settings)

        self.engine_tab = EngineTab(self.tabs, self.agent, self.save_settings)

        self.simulation_tab = SimulationTab(self.tabs, self) 
        self.tabs.add(self.simulation_tab, text="🧪 Simulation")

        # 4. 하단 컨트롤 패널
        self.create_bottom_panel()

        # 디버그 모드 토글 버튼
        self.chk_trace = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.frame_controls, 
            text="상세 추적(Trace) 켜기", 
            variable=self.chk_trace, 
            command=self.toggle_trace_mode
        ).pack(side="top", pady=5)

    def create_bottom_panel(self):
        # [수정] frame 지역 변수 대신 self.frame_controls 멤버 변수 사용
        # 그래야 setup_ui()에서 체크박스를 추가할 때 이 프레임을 찾을 수 있습니다.
        self.frame_controls = ttk.Frame(self.frame_right)
        self.frame_controls.pack(side="bottom", fill="x", padx=5, pady=10)
        
        # 아래의 모든 frame 참조를 self.frame_controls로 변경
        ttk.Button(self.frame_controls, text="🔍 메이플 창 찾기", command=self.find_window_action).pack(fill="x", pady=2)
        
        roi_frame = ttk.Frame(self.frame_controls)
        roi_frame.pack(fill="x", pady=2)
        ttk.Button(roi_frame, text="🎯 킬 카운트 영역", command=lambda: self.open_roi_selector("kill")).pack(side="left", fill="x", expand=True)
        ttk.Button(roi_frame, text="🗺️ 미니맵 영역", command=lambda: self.open_roi_selector("minimap")).pack(side="right", fill="x", expand=True)
        
        self.btn_record = ttk.Button(self.frame_controls, text="⏺ REC (데이터 녹화)", command=self.toggle_recording_action)
        self.btn_record.pack(fill="x", pady=5)
        
        self.btn_bot = ttk.Button(self.frame_controls, text="🤖 AUTO HUNT (봇 가동)", command=self.toggle_bot_action)
        self.btn_bot.pack(fill="x", ipady=10, pady=5)
        
        self.lbl_bot_status = ttk.Label(self.frame_controls, text="[BOT: OFF]", foreground="red", justify="center")
        self.lbl_bot_status.pack()

    def update_ui_loop(self):
        """화면 갱신 루프"""
        # 창이 닫혀버렸거나 소멸된 경우 루프 중단 (에러 방지)
        if not self.canvas.winfo_exists():
            return

        if self.is_simulating and self.sim_mode:
            # [시뮬레이션 모드]
            self.sim_mode.update()
            
        else:
            # [기존 게임 모드]
            debug_info = self.agent.get_debug_info()

            # -----------------------------------------------------------------
            # [★신규] 맵 제작 도구(MapCreator)의 데이터를 시각화 정보에 주입
            # -----------------------------------------------------------------
            if self.map_tab and hasattr(self.map_tab, 'map_creator'):
                creator = self.map_tab.map_creator
                
                # 시각화에 필요한 제작 도구 상태를 딕셔너리로 묶음
                creator_data = {
                    "new_platforms": creator.new_platforms,
                    "new_portals": getattr(creator, 'new_portals', []),
                    "new_ropes": getattr(creator, 'new_ropes', []),
                    "new_map_portals": getattr(creator, 'new_map_portals', []),
                    "temp_start": creator.temp_start_pos,
                    "temp_end": creator.temp_end_pos,
                

                    "selected_type": getattr(creator, 'selected_type', None),
                    "selected_index": getattr(creator, 'selected_index', None)
                }
                # debug_info에 'creator_data' 키로 추가
                debug_info['creator_data'] = creator_data
            # -----------------------------------------------------------------

            # [신규] 창 제목에 FPS 실시간 표시
            current_fps = debug_info.get("fps", 0.0)
            self.root.title(f"MapleHunter v2.0 - [FPS: {current_fps:.1f}]")
            
            # 1. 상태 패널 업데이트
            if self.status_panel:
                self.status_panel.update_stats(debug_info)

            # 2. 캔버스 그리기 (리사이징 적용)
            ox = self.map_tab.map_offset_x
            oy = self.map_tab.map_offset_y
            
            # 원본 OpenCV 이미지 생성 (수정된 Visualizer 호출)
            cv_img = Visualizer.draw_debug_view(debug_info, ox, oy)
            
            if cv_img is not None:
                w = self.canvas.winfo_width()
                h = self.canvas.winfo_height()
                
                if w > 1 and h > 1:
                    target_w = int(w * self.view_scale)
                    target_h = int(h * self.view_scale)
                    
                    tk_img = Visualizer.convert_to_tk_image(cv_img, target_w=target_w, target_h=target_h)
                    
                    if tk_img:
                        self.canvas.create_image(w//2, h//2, image=tk_img, anchor="center")
                        self.canvas.image = tk_img

        self.root.after(30, self.update_ui_loop)

    def find_window_action(self):
        if self.agent.vision.find_window():
            messagebox.showinfo("성공", "창을 찾았습니다.")
            self.status_panel.log("Game window found.")
        else:
            messagebox.showerror("실패", "창을 못 찾았습니다.")

    def toggle_bot_action(self):
        print(">>> [DEBUG] AUTO HUNT 버튼 클릭됨!") # 클릭 확인용 로그
        
        try:
            if self.agent.running:
                print(">>> [DEBUG] 봇 정지 요청")
                self.agent.stop()
                self.btn_bot.config(text="🤖 AUTO HUNT (봇 가동)")
                self.lbl_bot_status.config(text="[BOT: OFF]", foreground="red")
                if self.status_panel: self.status_panel.log("Bot stopped by user.")
            else:
                print(">>> [DEBUG] 봇 시작 요청")
                
                # [중요] 봇 시작 전 상태 체크
                if not self.agent.map_processor.platforms:
                    print(">>> [DEBUG] 경고: 맵 데이터가 없음")
                
                self.agent.start()
                self.agent.set_state('COMBAT') # 강제 전투 모드 진입
                
                print(f">>> [DEBUG] 봇 스레드 시작됨 (Running: {self.agent.running})")
                
                self.btn_bot.config(text="⏹ STOP BOT", state="normal")
                self.lbl_bot_status.config(text="[BOT: ON]", foreground="green")
                if self.status_panel: self.status_panel.log("Bot started.")
                
        except Exception as e:
            print(f">>> [CRITICAL ERROR] 봇 시작 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("오류", f"봇 시작 실패:\n{e}")

    def change_zoom(self, delta, reset=False):
        # 현재 활성화된 뷰포트 결정
        target_viewport = self.sim_mode.viewport if (self.is_simulating and self.sim_mode) else self.viewport
        
        if reset:
            target_viewport.zoom_scale = 1.0
            if self.is_simulating and self.sim_mode:
                target_viewport.zoom_scale = 4.0 # 시뮬레이션 기본값 복구
                target_viewport.center_view()
        else:
            target_viewport.adjust_zoom(delta)
            
        # 라벨 업데이트
        self.lbl_zoom.config(text=f"{int(target_viewport.zoom_scale * 100)}%")
        
        if self.is_simulating and self.sim_mode:
            self.sim_mode.draw()

    def open_roi_selector(self, target):
        if not self.agent.vision.window_found:
            messagebox.showwarning("경고", "먼저 창을 찾아주세요.")
            return
        ROISelector(self.root, self.agent, target)

    def toggle_recording_action(self):
        self.agent.toggle_recording()
        txt = "⏹ STOP (저장 중...)" if self.agent.is_recording else "⏺ REC (데이터 녹화)"
        self.btn_record.config(text=txt)

    def on_job_change_handler(self, new_job):
        if self.last_selected_job and self.last_selected_job != new_job:
            self.save_settings(job_name_override=self.last_selected_job)
        self.load_settings(job_name_override=new_job)
        self.last_selected_job = new_job

    def save_settings(self, job_name_override=None, **kwargs):
        """
        설정 저장 (kwargs를 통해 호출 출처에서 전달된 경로 정보 등도 처리)
        """
        # 외부에서 전달된 경로 정보가 있다면 내부 변수 업데이트
        if 'map_path' in kwargs: self.cur_map_path = kwargs['map_path']
        if 'model_path' in kwargs: self.cur_lstm_path = kwargs['model_path']
        if 'physics_path' in kwargs: self.cur_rf_path = kwargs['physics_path']

        # 직업 선택 콤보박스 참조 수정 (SkillTab 내부)
        target_job = job_name_override if job_name_override else self.skill_tab.combo_job.get()
        if not target_job: return

        data = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f: data = json.load(f)
            except: pass
        
        # 공통 설정 저장
        data["last_job"] = self.skill_tab.combo_job.get()
        
        # [★수정] MainWindow의 변수가 아니라 MapTab의 변수를 참조하도록 변경
        data["map_offset_x"] = self.map_tab.map_offset_x
        data["map_offset_y"] = self.map_tab.map_offset_y

        # ROI 설정 저장
        if self.agent.vision.minimap_roi:
            data["minimap_roi"] = self.agent.vision.minimap_roi
        if self.agent.vision.kill_roi:
            data["kill_roi"] = self.agent.vision.kill_roi
        if self.agent.vision.skill_rois:
            data["skill_rois"] = self.agent.vision.skill_rois

        # 파일 경로 저장
        data["last_map_path"] = self.cur_map_path
        data["last_lstm_path"] = self.cur_lstm_path
        data["last_rf_path"] = self.cur_rf_path

        if "job_settings" not in data: data["job_settings"] = {}
        
        # 스킬 설정 저장
        s_data = []
        for r in self.skill_tab.skill_rows:
            try:
                if r['frame'].winfo_exists():
                    s_data.append({
                        "name": r['name'].get(), 
                        "key": r['key'].get(), 
                        "cd": r['cd'].get(), 
                        "dur": r['dur'].get()
                    })
            except: pass
            
        i_data = []
        for r in self.skill_tab.install_rows:
            try:
                if r['frame'].winfo_exists():
                    i_data.append({
                        "name": r['name'].get(), 
                        "key": r['key'].get(), 
                        "range": r['range'].get(), 
                        "dur": r['dur'].get()
                    })
            except: pass
            
        data["job_settings"][target_job] = {"skills": s_data, "installs": i_data}

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            # 자동 저장(kwargs 호출)이 아닌 경우에만 메시지 표시
            if not job_name_override and not kwargs:
                messagebox.showinfo("저장", "설정이 저장되었습니다.")
                if self.status_panel: self.status_panel.log(f"Settings saved for {target_job}.")
                
        except Exception as e:
            print(f"저장 실패: {e}")

    def load_settings(self, job_name_override=None):
        if not os.path.exists(self.config_path): return
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            # [추가 1] 최근 맵 파일 자동 로드
            last_map = data.get("last_map_path", "")
            if last_map and os.path.exists(last_map):
                self.cur_map_path = last_map
                if self.agent.load_map(last_map):
                    print(f"✅ 자동 맵 로드 완료: {os.path.basename(last_map)}")
            
            # [추가 2] 최근 AI 모델(LSTM) 자동 로드
            last_lstm = data.get("last_lstm_path", "")
            if last_lstm and os.path.exists(last_lstm):
                self.cur_lstm_path = last_lstm
                if hasattr(self.agent, 'model_loader'):
                    # ModelLoader를 통해 모델 로드 시도
                    try:
                        self.agent.model_loader.load_model(last_lstm)
                        print(f"✅ 자동 AI 모델 로드 완료: {os.path.basename(last_lstm)}")
                    except Exception as e:
                        print(f"❌ AI 모델 자동 로드 실패: {e}")

            # [추가 3] 최근 물리 모델(Physics) 자동 로드
            last_rf = data.get("last_rf_path", "")
            if last_rf and os.path.exists(last_rf):
                self.cur_rf_path = last_rf
                if hasattr(self.agent, 'physics_engine'):
                    try:
                        self.agent.physics_engine.load_model(last_rf)
                        print(f"✅ 자동 물리 모델 로드 완료: {os.path.basename(last_rf)}")
                    except Exception as e:
                        print(f"❌ 물리 모델 자동 로드 실패: {e}")

            # --- 기존 설정 로드 로직 ---
            last_job = data.get("last_job", "")
            if not job_name_override and last_job:
                self.skill_tab.combo_job.set(last_job)
                self.last_selected_job = last_job
                
            self.map_tab.map_offset_x = data.get("map_offset_x", 0)
            self.map_tab.map_offset_y = data.get("map_offset_y", 0)
            self.agent.set_map_offset(self.map_tab.map_offset_x, self.map_tab.map_offset_y)            
            self.map_tab.adjust_offset(0, 0) 
            
            if data.get("minimap_roi"): self.agent.vision.set_minimap_roi(tuple(data["minimap_roi"]))
            if data.get("kill_roi"): self.agent.vision.set_roi(tuple(data["kill_roi"]))
            for n, i in data.get("skill_rois", {}).items():
                rect = tuple(i['rect'])
                thresh = i['threshold']
                
                # 1. VisionSystem에 등록 (기존 코드)
                self.agent.vision.set_skill_roi(n, rect, threshold=thresh)
                
                # 2. Scanner에 등록 (봇 판단용 - 추가해야 할 부분)
                if self.agent.scanner:
                    self.agent.scanner.register_skill(n, rect, threshold=thresh)

            target = job_name_override if job_name_override else last_job
            j_data = data.get("job_settings", {}).get(target, {})
            
            for r in self.skill_tab.skill_rows: r['frame'].destroy()
            self.skill_tab.skill_rows = []
            for r in self.skill_tab.install_rows: r['frame'].destroy()
            self.skill_tab.install_rows = []
            
            for s in j_data.get("skills", []):
                self.skill_tab.add_skill_row(s["name"], s["key"], s["cd"], s["dur"])
            for i in j_data.get("installs", []):
                self.skill_tab.add_install_row(i["name"], i["key"], i["range"], i["dur"])

            # 키 매핑 업데이트
            if self.agent:
                key_mapping = {}
                for r in self.skill_tab.skill_rows:
                    try:
                        name = r['name'].get()
                        key = r['key'].get()
                        if name and key:
                            key_mapping[name] = key.lower()
                    except: pass
                for r in self.skill_tab.install_rows:
                    try:
                        name = r['name'].get()
                        key = r['key'].get()
                        if name and key:
                            key_mapping[name] = key.lower()
                    except: pass
                self.agent.key_mapping = key_mapping
                print(f"BotAgent Key Mapping Updated: {key_mapping}")

                if self.agent.skill_strategy:
                    # 기존 정보 초기화
                    self.agent.skill_strategy.skills = {}
                    
                    target = job_name_override if job_name_override else last_job
                    j_data = data.get("job_settings", {}).get(target, {})

                    # 1. 일반 스킬 등록
                    for s in j_data.get("skills", []):
                        name = s["name"]
                        # 지속시간(dur)이 있으면 버프, 없으면 주력기(main)로 간주
                        try:
                            dur = float(s.get("dur", 0))
                            s_type = "buff" if dur > 0 else "main"
                            cd = float(s.get("cd", 0))
                        except:
                            s_type = "main"; cd = 0
                            
                        self.agent.skill_strategy.register_skill_info(name, s_type, cd)
                        print(f"전략 등록(Skill): {name} [{s_type}]")

                    # 2. 설치기 등록 (중요: 위치 로직 무시하고 쿨마다 쓰게 하려면 'buff'로 등록)
                    for i in j_data.get("installs", []):
                        name = i["name"]
                        # 설치기지만 'buff' 타입으로 등록하여 쿨타임마다 즉시 사용 유도
                        # (Scanner가 쿨타임을 관리하므로 여기서 CD 값은 크게 중요하지 않음)
                        self.agent.skill_strategy.register_skill_info(name, "buff", 0)
                        print(f"전략 등록(Install->Buff): {name}")

            if self.map_tab:
                self.map_tab.update_info(
                    map_path=self.cur_map_path,
                    lstm_path=self.cur_lstm_path,
                    rf_path=self.cur_rf_path
                )

            # EngineTab은 AI 모델 정보 업데이트
            if self.engine_tab:
                self.engine_tab.update_info(
                    lstm_path=self.cur_lstm_path,
                    rf_path=self.cur_rf_path
                )
                    
        except Exception as e:
            print(f"설정 로드 중 오류 발생: {e}")

    def toggle_trace_mode(self):
        from utils.logger import logger
        # 체크박스 상태에 따라 로거의 스위치를 켬/끔
        is_on = self.chk_trace.get()
        logger.set_tracing(is_on)

    def toggle_simulation_mode(self):
        """시뮬레이션 모드 켜기/끄기"""
        self.is_simulating = not self.is_simulating
        
        if self.is_simulating:
            # 시뮬레이션 모드 객체 생성 (초기화)
            if not self.sim_mode:
                self.sim_mode = SimulationMode(self)
            self.sim_mode.start()
            self.canvas.config(bg="#222222") # 배경색 변경으로 모드 구분
            self.root.title("MapleHunter v2.0 - [SIMULATION MODE]")
        else:
            if self.sim_mode:
                self.sim_mode.stop()
            self.canvas.delete("sim_obj") # 시뮬레이션 객체 삭제
            self.canvas.config(bg="black")
            
        return self.is_simulating
    
    def on_canvas_drag(self, event):
        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        self.last_mouse_pos = (event.x, event.y)
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # [수정] 모드에 따라 제어할 뷰포트 선택
        if self.is_simulating and self.sim_mode:
            self.sim_mode.viewport.pan_move(dx, dy, cw, ch)
            self.sim_mode.draw()
        else:
            self.viewport.pan_move(dx, dy, cw, ch)

    def on_mouse_wheel(self, event):
        delta = 0.2 if event.delta > 0 else -0.2
        
        # [수정] 모드에 따라 제어할 뷰포트 선택
        if self.is_simulating and self.sim_mode:
            self.sim_mode.viewport.adjust_zoom(delta)
            self.sim_mode.draw()
        else:
            self.viewport.adjust_zoom(delta)

    
