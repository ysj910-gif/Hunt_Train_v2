# ui/tabs/engine_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

class EngineTab:
    def __init__(self, notebook, agent, save_callback=None):
        self.agent = agent
        self.save_callback = save_callback
        
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Engine & AI")
        
        self._setup_ui()

    def _setup_ui(self):
        # AI Models (LSTM + Physics) - 기존 MapTab에서 가져옴
        model_frame = ttk.LabelFrame(self.frame, text="AI Models Configuration")
        model_frame.pack(fill="x", padx=10, pady=10)
        
        # 1. LSTM Model
        self.lbl_lstm = ttk.Label(model_frame, text="LSTM: Not Loaded", foreground="gray")
        self.lbl_lstm.pack(pady=(5, 2))
        ttk.Button(model_frame, text="🧠 Load LSTM Model (.pth)", command=self.load_lstm).pack(fill="x", padx=5, pady=5)

        # 구분선
        ttk.Separator(model_frame, orient='horizontal').pack(fill='x', pady=10)

        # 2. Physics Engine
        self.lbl_physics = ttk.Label(model_frame, text="Physics: Not Loaded", foreground="gray")
        self.lbl_physics.pack(pady=(5, 2))
        ttk.Button(model_frame, text="⚛️ Load Physics Engine (.pth)", command=self.load_physics_model).pack(fill="x", padx=5, pady=5)

    def load_lstm(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            if self.agent.model_loader.load_model(path):
                self.lbl_lstm.config(text=f"LSTM: {os.path.basename(path)}", foreground="blue")
                # 설정 저장 콜백 호출
                if self.save_callback: 
                    self.save_callback(model_path=path)
            else:
                messagebox.showerror("Error", "LSTM 모델 로드 실패")

    def load_physics_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            if hasattr(self.agent, 'physics_engine') and self.agent.physics_engine:
                if self.agent.physics_engine.load_model(path):
                    self.lbl_physics.config(text=f"Physics: {os.path.basename(path)}", foreground="blue")
                    # 설정 저장 콜백 호출
                    if self.save_callback: 
                        self.save_callback(physics_path=path)
                    return
            
            messagebox.showerror("Error", "물리 엔진 로드 실패\n(BotAgent 초기화를 확인하세요)")

    def update_info(self, lstm_path=None, rf_path=None):
        """외부 설정 파일 로드 시 UI 업데이트"""
        if lstm_path and os.path.exists(lstm_path):
            self.lbl_lstm.config(text=f"LSTM: {os.path.basename(lstm_path)}", foreground="blue")

        if rf_path and os.path.exists(rf_path):
            self.lbl_physics.config(text=f"Physics: {os.path.basename(rf_path)}", foreground="blue")