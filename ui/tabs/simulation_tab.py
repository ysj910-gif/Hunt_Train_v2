# ui/tabs/simulation_tab.py

import tkinter as tk
from tkinter import ttk

class SimulationTab(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.mw = main_window # MainWindow 참조
        self.setup_ui()

    def setup_ui(self):
        # 1. 모드 활성화 스위치
        lf_mode = ttk.LabelFrame(self, text="Mode Control")
        lf_mode.pack(fill="x", padx=5, pady=5)
        
        self.btn_toggle = ttk.Button(lf_mode, text="▶ Start Simulation", command=self.toggle_simulation)
        self.btn_toggle.pack(fill="x", padx=5, pady=5)

        # 2. 위치 제어
        lf_pos = ttk.LabelFrame(self, text="Set Position")
        lf_pos.pack(fill="x", padx=5, pady=5)
        
        f_xy = ttk.Frame(lf_pos)
        f_xy.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(f_xy, text="X:").pack(side="left")
        self.ent_x = ttk.Entry(f_xy, width=6)
        self.ent_x.insert(0, "125")
        self.ent_x.pack(side="left", padx=2)
        
        ttk.Label(f_xy, text="Y:").pack(side="left")
        self.ent_y = ttk.Entry(f_xy, width=6)
        self.ent_y.insert(0, "66")
        self.ent_y.pack(side="left", padx=2)
        
        ttk.Button(lf_pos, text="Teleport", command=self.teleport).pack(fill="x", padx=5, pady=2)

        # 3. 수동 조작 (물리 엔진 테스트)
        lf_manual = ttk.LabelFrame(self, text="Manual Input (Physics)")
        lf_manual.pack(fill="x", padx=5, pady=5)
        
        f_move = ttk.Frame(lf_manual)
        f_move.pack(fill="x")
        ttk.Button(f_move, text="⬅️", command=lambda: self.input_action("move_left")).pack(side="left", expand=True, fill="x")
        ttk.Button(f_move, text="➡️", command=lambda: self.input_action("move_right")).pack(side="left", expand=True, fill="x")
        
        ttk.Button(lf_manual, text="Jump (Alt)", command=lambda: self.input_action("jump")).pack(fill="x", padx=5, pady=2)
        ttk.Button(lf_manual, text="Up Jump (↑+Alt)", command=lambda: self.input_action("up_jump")).pack(fill="x", padx=5, pady=2)

        # 4. 봇 테스트
        lf_bot = ttk.LabelFrame(self, text="Bot AI Test")
        lf_bot.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(lf_bot, text="🤖 Run Next Step", command=self.run_bot_step).pack(fill="x", padx=5, pady=5)
        self.lbl_bot_log = ttk.Label(lf_bot, text="Ready", wraplength=200)
        self.lbl_bot_log.pack(pady=5)

    def toggle_simulation(self):
        # MainWindow의 시뮬레이션 모드 토글 메서드 호출
        is_active = self.mw.toggle_simulation_mode()
        if is_active:
            self.btn_toggle.config(text="⏹ Stop Simulation")
        else:
            self.btn_toggle.config(text="▶ Start Simulation")

    def teleport(self):
        try:
            x = float(self.ent_x.get())
            y = float(self.ent_y.get())
            if self.mw.sim_mode:
                self.mw.sim_mode.char_x = x
                self.mw.sim_mode.char_y = y
                self.mw.sim_mode.vx = 0
                self.mw.sim_mode.vy = 0
        except ValueError:
            pass

    def input_action(self, action):
        if self.mw.sim_mode and self.mw.sim_mode.active:
            self.mw.sim_mode.apply_action(action)

    def run_bot_step(self):
        # 봇의 판단 로직을 시뮬레이터 상태 기준으로 실행
        if not self.mw.sim_mode: return
        
        sim = self.mw.sim_mode
        current_pos = (sim.char_x, sim.char_y)
        
        # 설치기 정보 등은 임시로 설정
        install_ready = {"fountain": True}
        
        # PathFinder에게 질문
        cmd, target = sim.path_finder.get_next_combat_step(current_pos, install_ready)
        
        self.lbl_bot_log.config(text=f"Cmd: {cmd}\nTarget: {target}")
        
        # 봇이 결정한 행동을 바로 물리 엔진에 적용하려면:
        if cmd == "execute_path":
            sim.apply_action(target) # target이 'up_jump' 같은 문자열임
        elif cmd == "move_to_install":
            # 걷기 방향 결정
            direction = "move_right" if target[0] > sim.char_x else "move_left"
            sim.apply_action(direction)