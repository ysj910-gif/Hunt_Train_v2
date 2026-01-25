# ui/status_panel.py
import tkinter as tk
from tkinter import ttk
import datetime

class StatusPanel:
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="System Status & Logs")
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 1. 상단: 대시보드
        self.dashboard_frame = ttk.Frame(self.frame)
        self.dashboard_frame.pack(fill="x", pady=5)
        
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Arial", 11, "bold")) # 폰트 약간 조정
        
        # 라벨 생성 (배치 순서 조정)
        self.lbl_state = ttk.Label(self.dashboard_frame, text="ST: IDLE", style="Big.TLabel", foreground="blue")
        self.lbl_state.pack(side="left", padx=5)
        
        self.lbl_kill = ttk.Label(self.dashboard_frame, text="Kill: 0", style="Big.TLabel", foreground="red")
        self.lbl_kill.pack(side="left", padx=5)

        self.lbl_plat = ttk.Label(self.dashboard_frame, text="Plat: -", style="Big.TLabel", foreground="green")
        self.lbl_plat.pack(side="left", padx=5)
        
        self.lbl_pos = ttk.Label(self.dashboard_frame, text="Pos: (0,0)", font=("Consolas", 10))
        self.lbl_pos.pack(side="left", padx=5)

        # 2. 하단: 로그 콘솔
        self.console_frame = ttk.Frame(self.frame)
        self.console_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(self.console_frame, height=8, state="disabled", bg="#f0f0f0", font=("Consolas", 9))
        self.scrollbar = ttk.Scrollbar(self.console_frame, orient="vertical", command=self.log_text.yview)
        
        self.log_text.configure(yscrollcommand=self.scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.log("System Ready.")

    def update_stats(self, debug_info):
        """Agent 정보를 받아 UI 갱신"""
        state = debug_info.get("state", "-")
        # action = debug_info.get("action", "-") # 공간 부족시 생략 가능
        pos = debug_info.get("player_pos", (0,0))
        kill = debug_info.get("kill_count", 0)
        p_idx = debug_info.get("current_plat_idx", -1)
        
        self.lbl_state.config(text=f"ST: {state}")
        self.lbl_kill.config(text=f"Kill: {kill}")
        self.lbl_plat.config(text=f"Plat: #{p_idx}" if p_idx != -1 else "Plat: None")
        self.lbl_pos.config(text=f"Pos: {pos}")
        
        # 필요시 여기에 킬 카운트나 시간 정보 추가 가능

    def log(self, message):
        """로그 창에 메시지 추가"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state="normal")
        self.log_text.insert("end", full_msg)
        self.log_text.see("end") # 자동 스크롤
        self.log_text.config(state="disabled")