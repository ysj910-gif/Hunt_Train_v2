# ui/status_panel.py
import tkinter as tk
from tkinter import ttk, filedialog
import datetime
import os
import json
import re
from core.latency_monitor import latency_monitor

class StatusPanel:
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="System Status & Logs")
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 1. 상단: 대시보드
        self.dashboard_frame = ttk.Frame(self.frame)
        self.dashboard_frame.pack(fill="x", pady=5)
        
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Arial", 11, "bold"))
        
        # 상태 표시 라벨들
        self.lbl_state = ttk.Label(self.dashboard_frame, text="ST: IDLE", style="Big.TLabel", foreground="blue")
        self.lbl_state.pack(side="left", padx=5)
        
        self.lbl_kill = ttk.Label(self.dashboard_frame, text="Kill: 0", style="Big.TLabel", foreground="red")
        self.lbl_kill.pack(side="left", padx=5)

        self.lbl_plat = ttk.Label(self.dashboard_frame, text="Plat: -", style="Big.TLabel", foreground="green")
        self.lbl_plat.pack(side="left", padx=5)
        
        self.lbl_pos = ttk.Label(self.dashboard_frame, text="Pos: (0,0)", font=("Consolas", 10))
        self.lbl_pos.pack(side="left", padx=5)

        self.lbl_latency = ttk.Label(self.dashboard_frame, text="Lat: -ms", font=("Consolas", 10), foreground="purple")
        self.lbl_latency.pack(side="left", padx=5)

        # 컨트롤 버튼 영역 (우측 정렬)
        self.ctrl_frame = ttk.Frame(self.dashboard_frame)
        self.ctrl_frame.pack(side="right", padx=5)

        # 버튼 0: 로그 파일 열기
        self.btn_load = ttk.Button(self.ctrl_frame, text="📂 로그 열기", command=self.open_log_file, width=10)
        self.btn_load.pack(side="right", padx=2)

        # 버튼 1: 정렬 토글
        self.sort_mode = "asc" # asc: 과거순(기본), desc: 최신순
        self.btn_sort = ttk.Button(self.ctrl_frame, text="▼ 최신순", command=self.toggle_sort, width=8)
        self.btn_sort.pack(side="right", padx=2)

        # 버튼 2: 에러 필터
        self.filter_error = False
        self.btn_filter = ttk.Button(self.ctrl_frame, text="⚠ 에러만", command=self.toggle_filter, width=8)
        self.btn_filter.pack(side="right", padx=2)

        # 버튼 3: 최근 1분 필터
        self.filter_recent = False
        self.btn_recent = ttk.Button(self.ctrl_frame, text="⏱ 최근1분", command=self.toggle_recent, width=8)
        self.btn_recent.pack(side="right", padx=2)

        # 버튼 4: 로그 강조 (오류가 발생했던 부분)
        self.btn_trace = ttk.Button(self.ctrl_frame, text="🔍 강조", command=self.highlight_recent_logs, width=6)
        self.btn_trace.pack(side="right", padx=2)

        # 2. 하단: 로그 콘솔
        self.console_frame = ttk.Frame(self.frame)
        self.console_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(self.console_frame, height=8, state="disabled", bg="#f0f0f0", font=("Consolas", 9))
        self.scrollbar = ttk.Scrollbar(self.console_frame, orient="vertical", command=self.log_text.yview)
        
        self.log_text.configure(yscrollcommand=self.scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.log_text.tag_config("highlight", background="#ffff99", foreground="red")
        
        self.log_history = [] 
        self.log("System Ready.")

    def update_stats(self, debug_info):
        """Agent 정보를 받아 UI 갱신"""
        state = debug_info.get("state", "-")
        pos = debug_info.get("player_pos", (0,0))
        kill = debug_info.get("kill_count", 0)
        p_idx = debug_info.get("current_plat_idx", -1)
        
        self.lbl_state.config(text=f"ST: {state}")
        self.lbl_kill.config(text=f"Kill: {kill}")
        self.lbl_plat.config(text=f"Plat: #{p_idx}" if p_idx != -1 else "Plat: None")
        self.lbl_pos.config(text=f"Pos: {pos}")

        # [NEW] Latency 정보 갱신
        lat_str = latency_monitor.get_latency_info()
        self.lbl_latency.config(text=f"Lat: {lat_str}")

    def log(self, message):
        """실시간 로그 추가"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        
        is_err = any(k in message.lower() for k in ['error', 'exception', 'fail', 'critical'])
        self.log_history.append({
            'timestamp': datetime.datetime.now(),
            'msg': full_msg,
            'is_error': is_err
        })
        
        self._append_log_to_view(self.log_history[-1])

    def _append_log_to_view(self, entry):
        """필터 조건에 맞으면 뷰에 추가"""
        # 에러 필터
        if self.filter_error and not entry['is_error']:
            return
            
        # 최근 1분 필터 (기준: 마지막 로그 시간)
        if self.filter_recent and self.log_history:
            last_time = self.log_history[-1]['timestamp']
            cutoff_time = last_time - datetime.timedelta(seconds=60)
            if entry['timestamp'] < cutoff_time:
                return

        self.log_text.config(state="normal")
        if self.sort_mode == "desc":
            self.log_text.insert("1.0", entry['msg'] + "\n")
        else:
            self.log_text.insert("end", entry['msg'] + "\n")
            self.log_text.see("end")
        self.log_text.config(state="disabled")

    # --- [신규 기능] 로그 파일 로드 및 파싱 ---
    def open_log_file(self):
        log_dir = r"C:\Temp\logs"
        if not os.path.exists(log_dir): log_dir = os.getcwd()
        
        filepath = filedialog.askopenfilename(
            initialdir=log_dir,
            title="로그 파일 선택 (System / Decision)",
            filetypes=[("Log Files", "*.log *.jsonl"), ("All Files", "*.*")]
        )
        if not filepath: return

        # 기존 로그 비우기
        self.log_history = []
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        filename = os.path.basename(filepath)
        
        # 파일 타입에 따라 파싱 분기
        if filename.endswith(".jsonl") or "decision" in filename:
            self._parse_decision_log(filepath)
        else:
            self._parse_system_log(filepath)
            
        # 화면 갱신
        self.refresh_view()
        # 안내 메시지 (기록엔 남기지 않음)
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"\n=== 파일 로드 완료: {filename} ({len(self.log_history)} 라인) ===\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _parse_system_log(self, filepath):
        """system_yyyy-mm-dd.log 파싱"""
        file_date = datetime.date.today()
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath)
        if date_match:
            try:
                file_date = datetime.datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
            except: pass

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                match = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", line)
                dt = datetime.datetime.now()
                if match:
                    t_str = match.group(1)
                    t_obj = datetime.datetime.strptime(t_str, "%H:%M:%S").time()
                    dt = datetime.datetime.combine(file_date, t_obj)
                
                is_err = any(k in line.upper() for k in ['ERROR', 'CRITICAL', 'EXCEPTION'])
                self.log_history.append({'timestamp': dt, 'msg': line, 'is_error': is_err})

    def _parse_decision_log(self, filepath):
        """decision_history.jsonl 파싱"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    entry = json.loads(line)
                    ts_str = entry.get('timestamp', '')
                    try:
                        dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    except:
                        dt = datetime.datetime.now()
                        
                    step = entry.get('step', '')
                    state = entry.get('state', '')
                    decision = entry.get('decision', '')
                    reason = entry.get('reason', '')
                    
                    msg = f"[{ts_str}] [{step}] {state} -> {decision} ({reason})"
                    is_err = 'FAIL' in decision.upper() or 'ERROR' in state.upper()
                    
                    self.log_history.append({'timestamp': dt, 'msg': msg, 'is_error': is_err})
                except: continue

    # --- 컨트롤 기능 (토글) ---
    def toggle_sort(self):
        if self.sort_mode == "asc":
            self.sort_mode = "desc"
            self.btn_sort.config(text="▲ 과거순")
        else:
            self.sort_mode = "asc"
            self.btn_sort.config(text="▼ 최신순")
        self.refresh_view()

    def toggle_filter(self):
        self.filter_error = not self.filter_error
        self.btn_filter.config(text="전체 보기" if self.filter_error else "⚠ 에러만")
        self.refresh_view()

    def toggle_recent(self):
        self.filter_recent = not self.filter_recent
        self.btn_recent.config(text="전체 보기" if self.filter_recent else "⏱ 최근1분")
        self.refresh_view()

    def refresh_view(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        
        is_reverse = (self.sort_mode == "desc")
        sorted_logs = sorted(self.log_history, key=lambda x: x['timestamp'], reverse=is_reverse)
        
        cutoff_time = None
        if self.log_history:
            last_time = self.log_history[-1]['timestamp']
            cutoff_time = last_time - datetime.timedelta(seconds=60)
        
        for entry in sorted_logs:
            if self.filter_error and not entry['is_error']:
                continue
            
            if self.filter_recent and cutoff_time and entry['timestamp'] < cutoff_time:
                continue

            self.log_text.insert("end", entry['msg'] + "\n")
            
        self.log_text.config(state="disabled")
        
        if self.sort_mode == "desc":
            self.log_text.see("1.0")
        else:
            self.log_text.see("end")

    def highlight_recent_logs(self):
        self.log_text.config(state="normal")
        self.log_text.tag_remove("highlight", "1.0", "end")
        
        if not self.log_history: return
        
        ref_time = self.log_history[-1]['timestamp']
        cutoff_time = ref_time - datetime.timedelta(seconds=60)
        
        content = self.log_text.get("1.0", "end-1c")
        lines = content.split('\n')
        
        first_idx = None
        for i, line in enumerate(lines):
            if not line.strip(): continue
            try:
                match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                if match:
                    time_str = match.group(1)
                    log_time = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
                    log_dt = datetime.datetime.combine(ref_time.date(), log_time)
                    
                    if log_dt > ref_time + datetime.timedelta(minutes=10):
                        log_dt -= datetime.timedelta(days=1)
                    
                    if log_dt >= cutoff_time:
                        start = f"{i+1}.0"
                        end = f"{i+1}.end"
                        self.log_text.tag_add("highlight", start, end)
                        if first_idx is None: first_idx = start
            except: continue
        
        if first_idx: self.log_text.see(first_idx)
        self.log_text.config(state="disabled")