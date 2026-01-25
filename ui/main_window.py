# ui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

# 모듈 임포트
from modules.job_manager import JobManager
from ui.roi_selector import ROISelector
from ui.visualizer import Visualizer
from ui.tabs.skill_tab import SkillTab
from ui.tabs.map_tab import MapTab
from ui.status_panel import StatusPanel  # [신규 추가]

class MainWindow:
    def __init__(self, root, agent):
        self.root = root
        self.agent = agent
        
        self.root.title("MapleHunter v2.0 Modular UI")
        self.root.geometry("1300x900") # 너비 약간 확장
        
        self.job_mgr = JobManager()
        self.config_path = "config.json"
        
        self.skill_tab = None
        self.map_tab = None
        self.status_panel = None # [신규]
        
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
        self.map_tab = MapTab(self.tabs, self.agent)

        # 4. 하단 컨트롤 패널
        self.create_bottom_panel()

    def create_bottom_panel(self):
        # ... (기존 코드와 동일) ...
        frame = ttk.Frame(self.frame_right)
        frame.pack(side="bottom", fill="x", padx=5, pady=10)
        
        ttk.Button(frame, text="🔍 메이플 창 찾기", command=self.find_window_action).pack(fill="x", pady=2)
        
        roi_frame = ttk.Frame(frame)
        roi_frame.pack(fill="x", pady=2)
        ttk.Button(roi_frame, text="🎯 킬 카운트 영역", command=lambda: self.open_roi_selector("kill")).pack(side="left", fill="x", expand=True)
        ttk.Button(roi_frame, text="🗺️ 미니맵 영역", command=lambda: self.open_roi_selector("minimap")).pack(side="right", fill="x", expand=True)
        
        self.btn_record = ttk.Button(frame, text="⏺ REC (데이터 녹화)", command=self.toggle_recording_action)
        self.btn_record.pack(fill="x", pady=5)
        
        self.btn_bot = ttk.Button(frame, text="🤖 AUTO HUNT (봇 가동)", command=self.toggle_bot_action)
        self.btn_bot.pack(fill="x", ipady=10, pady=5)
        self.lbl_bot_status = ttk.Label(frame, text="[BOT: OFF]", foreground="red", justify="center")
        self.lbl_bot_status.pack()

    def update_ui_loop(self):
        """화면 갱신 루프"""
        # 창이 닫혀버렸거나 소멸된 경우 루프 중단 (에러 방지)
        if not self.canvas.winfo_exists():
            return

        debug_info = self.agent.get_debug_info()
        
        # 1. 상태 패널 업데이트
        if self.status_panel:
            self.status_panel.update_stats(debug_info)

        # 2. 캔버스 그리기 (리사이징 적용)
        ox = self.map_tab.map_offset_x
        oy = self.map_tab.map_offset_y
        
        # 원본 OpenCV 이미지 생성
        cv_img = Visualizer.draw_debug_view(debug_info, ox, oy)
        
        if cv_img is not None:
            # ★ 캔버스의 현재 크기 가져오기
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            
            # [수정] 창이 초기화되어 크기가 1보다 클 때만 그리기 수행
            if w > 1 and h > 1:
                # 캔버스 크기에 맞춰 비율 유지하며 리사이징된 Tk 이미지 변환
                tk_img = Visualizer.convert_to_tk_image(cv_img, target_w=w, target_h=h)
                
                if tk_img:
                    # 캔버스 중앙에 배치
                    self.canvas.create_image(w//2, h//2, image=tk_img, anchor="center")
                    self.canvas.image = tk_img # GC 방지

        self.root.after(30, self.update_ui_loop)

    # ... (나머지 핸들러 메서드들은 기존 코드 그대로 유지) ...
    def find_window_action(self):
        if self.agent.vision.find_window():
            messagebox.showinfo("성공", "창을 찾았습니다.")
            self.status_panel.log("Game window found.")
        else:
            messagebox.showerror("실패", "창을 못 찾았습니다.")

    def toggle_bot_action(self):
        if self.agent.running:
            self.agent.stop()
            self.btn_bot.config(text="🤖 AUTO HUNT (봇 가동)")
            self.lbl_bot_status.config(text="[BOT: OFF]", foreground="red")
            self.status_panel.log("Bot stopped by user.")
        else:
            self.agent.start()
            self.btn_bot.config(text="⏹ STOP BOT", state="normal")
            self.lbl_bot_status.config(text="[BOT: ON]", foreground="green")
            self.status_panel.log("Bot started.")

    # ... (open_roi_selector, toggle_recording_action 등 기존 유지) ...
    # (코드 중략: 기존 메서드들은 변경 사항 없음)
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

    def save_settings(self, job_name_override=None):
        # (기존 save_settings 코드 그대로 사용)
        target_job = job_name_override if job_name_override else self.skill_tab.combo_job.get()
        if not target_job: return

        data = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f: data = json.load(f)
            except: pass
        
        data["last_job"] = self.skill_tab.combo_job.get()
        data["map_offset_x"] = self.map_tab.map_offset_x
        data["map_offset_y"] = self.map_tab.map_offset_y
        data["minimap_roi"] = self.agent.vision.minimap_roi
        data["kill_roi"] = self.agent.vision.kill_roi
        data["skill_rois"] = self.agent.vision.skill_rois

        if "job_settings" not in data: data["job_settings"] = {}
        
        s_data = []
        for r in self.skill_tab.skill_rows:
            try:
                if r['frame'].winfo_exists():
                    s_data.append({"name": r['name'].get(), "key": r['key'].get(), "cd": r['cd'].get(), "dur": r['dur'].get()})
            except: pass
            
        i_data = []
        for r in self.skill_tab.install_rows:
            try:
                if r['frame'].winfo_exists():
                    i_data.append({"name": r['name'].get(), "key": r['key'].get(), "range": r['range'].get(), "dur": r['dur'].get()})
            except: pass
            
        data["job_settings"][target_job] = {"skills": s_data, "installs": i_data}

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            if not job_name_override:
                messagebox.showinfo("저장", "설정이 저장되었습니다.")
                self.status_panel.log(f"Settings saved for {target_job}.")
        except Exception as e:
            print(f"저장 실패: {e}")

    def load_settings(self, job_name_override=None):
        # (기존 load_settings 코드 그대로 사용)
        if not os.path.exists(self.config_path): return
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            last_job = data.get("last_job", "")
            if not job_name_override and last_job:
                self.skill_tab.combo_job.set(last_job)
                self.last_selected_job = last_job
                
            self.map_tab.map_offset_x = data.get("map_offset_x", 0)
            self.map_tab.map_offset_y = data.get("map_offset_y", 0)
            self.map_tab.adjust_offset(0, 0)
            
            if data.get("minimap_roi"): self.agent.vision.set_minimap_roi(tuple(data["minimap_roi"]))
            if data.get("kill_roi"): self.agent.vision.set_roi(tuple(data["kill_roi"]))
            for n, i in data.get("skill_rois", {}).items():
                self.agent.vision.set_skill_roi(n, tuple(i['rect']), threshold=i['threshold'])
            
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
                
        except Exception as e:
            print(f"로드 오류: {e}")