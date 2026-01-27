# ui/tabs/skill_tab.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from ui.components.roi_selector import ROISelector

class SkillTab:
    def __init__(self, notebook, agent, job_mgr, save_callback):
        self.agent = agent
        self.job_mgr = job_mgr
        self.save_callback = save_callback # 저장 시 Main의 저장 함수 호출
        
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Skills & Info")
        
        self.skill_rows = []
        self.install_rows = []
        
        self._setup_ui()

    def _setup_ui(self):
        # 1. 직업 선택 영역
        job_frame = ttk.LabelFrame(self.frame, text="Player Info")
        job_frame.pack(fill="x", pady=5)
        
        ttk.Label(job_frame, text="Job Class:").pack(side="left", padx=5)
        
        job_list = self.job_mgr.get_all_jobs()
        self.combo_job = ttk.Combobox(job_frame, values=job_list, state="readonly")
        self.combo_job.pack(side="left", fill="x", expand=True, padx=5)
        if job_list: self.combo_job.current(0)
        self.combo_job.bind("<<ComboboxSelected>>", self.on_job_change)

        ttk.Button(job_frame, text="+", width=3, command=self.add_job_action).pack(side="left", padx=2)

        # 2. 일반 스킬 영역
        self._create_scrollable_area("Active Skills (Buff/Attack)", self.add_skill_row)

        # 3. 설치기 영역
        install_frame = ttk.LabelFrame(self.frame, text="Install Skills (Map Coverage)")
        install_frame.pack(fill="x", pady=5)
        
        # 헤더
        ih_frame = ttk.Frame(install_frame)
        ih_frame.pack(fill="x")
        for t, w in [("Name", 8), ("Key", 4), ("U/D/L/R", 12), ("Dur", 4)]:
            ttk.Label(ih_frame, text=t, width=w).pack(side="left", padx=1)
            
        self.install_list_frame = ttk.Frame(install_frame)
        self.install_list_frame.pack(fill="x")
        ttk.Button(install_frame, text="+ Add Install Skill", command=self.add_install_row).pack(fill="x", pady=2)

        # 저장 버튼
        ttk.Button(self.frame, text="💾 Save Config (All)", command=self.save_callback).pack(fill="x", pady=10)

    def _create_scrollable_area(self, title, add_cmd):
        frame = ttk.LabelFrame(self.frame, text=title)
        frame.pack(fill="both", expand=True, pady=5)
        
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.skill_list_frame = ttk.Frame(canvas)
        
        canvas.create_window((0, 0), window=self.skill_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.skill_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # 스킬 헤더
        h_frame = ttk.Frame(self.skill_list_frame)
        h_frame.pack(fill="x")
        for t, w in [("Name", 12), ("Key", 5), ("CD(s)", 5), ("Dur(s)", 5)]:
            ttk.Label(h_frame, text=t, width=w).pack(side="left", padx=1)
        ttk.Button(h_frame, text="+", width=3, command=add_cmd).pack(side="left", padx=5)

    # --- Actions ---
    def add_skill_row(self, name="", key="", cd="0", dur="0"):
        row = ttk.Frame(self.skill_list_frame)
        row.pack(fill="x", pady=2)
        e_name = ttk.Entry(row, width=12); e_name.insert(0, name); e_name.pack(side="left")
        e_key = ttk.Entry(row, width=5); e_key.insert(0, key); e_key.pack(side="left")
        e_cd = ttk.Entry(row, width=5); e_cd.insert(0, cd); e_cd.pack(side="left")
        e_dur = ttk.Entry(row, width=5); e_dur.insert(0, dur); e_dur.pack(side="left")
        ttk.Button(row, text="X", width=2, command=lambda: self._delete_row(row, self.skill_rows)).pack(side="left")
        self.skill_rows.append({'frame': row, 'name': e_name, 'key': e_key, 'cd': e_cd, 'dur': e_dur})

    def add_install_row(self, name="", key="", udlr="0,0,0,0", dur="0"):
        row = ttk.Frame(self.install_list_frame)
        row.pack(fill="x", pady=2)
        e_name = ttk.Entry(row, width=8); e_name.insert(0, name); e_name.pack(side="left")
        e_key = ttk.Entry(row, width=4); e_key.insert(0, key); e_key.pack(side="left")
        e_range = ttk.Entry(row, width=12); e_range.insert(0, udlr); e_range.pack(side="left")
        e_dur = ttk.Entry(row, width=4); e_dur.insert(0, dur); e_dur.pack(side="left")
        
        # ROI 선택 버튼
        ttk.Button(row, text="👁️", width=3, 
                   command=lambda: self._open_roi_selector(e_name.get())).pack(side="left", padx=2)
        
        ttk.Button(row, text="X", width=2, command=lambda: self._delete_row(row, self.install_rows)).pack(side="left")
        self.install_rows.append({'frame': row, 'name': e_name, 'key': e_key, 'range': e_range, 'dur': e_dur})

    def add_job_action(self):
        new_job = simpledialog.askstring("직업 추가", "새로운 직업 이름을 입력하세요:")
        if new_job and new_job.strip():
            self.job_mgr.get_job_id(new_job)
            vals = list(self.combo_job['values'])
            if new_job not in vals:
                vals.append(new_job)
                self.combo_job['values'] = vals
            self.combo_job.set(new_job)
            self.on_job_change()

    def on_job_change(self, event=None):
        # Main Window의 저장/로드 로직 호출 필요 (구조상 Main에서 처리하거나 콜백)
        if hasattr(self, 'on_job_change_callback'):
            self.on_job_change_callback(self.combo_job.get())

    def _delete_row(self, row, list_ref):
        row.destroy()
        list_ref[:] = [r for r in list_ref if r['frame'] != row]

    def _open_roi_selector(self, skill_name):
        if not self.agent.vision.window_found:
            messagebox.showwarning("경고", "먼저 창을 찾아주세요.")
            return
        ROISelector(self.frame, self.agent, "skill", skill_name)