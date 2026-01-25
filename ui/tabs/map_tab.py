# ui/tabs/map_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

class MapTab:
    def __init__(self, notebook, agent):
        self.agent = agent
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Map & AI Model")
        
        self.map_offset_x = 0
        self.map_offset_y = 0
        
        self._setup_ui()

    def _setup_ui(self):
        # 1. 맵 로드
        map_frame = ttk.LabelFrame(self.frame, text="Map Data (.json)")
        map_frame.pack(fill="x", pady=5)
        self.lbl_map = ttk.Label(map_frame, text="No Map Loaded", foreground="gray")
        self.lbl_map.pack(pady=2)
        ttk.Button(map_frame, text="📂 Load Map JSON", command=self.load_map_file).pack(fill="x", padx=5, pady=5)
        
        # 2. 오프셋 조정
        offset_frame = ttk.LabelFrame(self.frame, text="Position Offset Correction")
        offset_frame.pack(fill="x", pady=5)
        self.lbl_offset = ttk.Label(offset_frame, text="Offset: (0, 0)", font=("Arial", 10, "bold"))
        self.lbl_offset.pack(pady=2)
        
        btn_pad = ttk.Frame(offset_frame)
        btn_pad.pack(pady=2)
        # 방향키 배치
        ttk.Button(btn_pad, text="▲", width=3, command=lambda: self.adjust_offset(0, -1)).grid(row=0, column=1)
        ttk.Button(btn_pad, text="◀", width=3, command=lambda: self.adjust_offset(-1, 0)).grid(row=1, column=0)
        ttk.Button(btn_pad, text="▼", width=3, command=lambda: self.adjust_offset(0, 1)).grid(row=1, column=1)
        ttk.Button(btn_pad, text="▶", width=3, command=lambda: self.adjust_offset(1, 0)).grid(row=1, column=2)
        ttk.Button(offset_frame, text="Reset", command=lambda: self.adjust_offset(0, 0, reset=True)).pack(pady=2)

        # 3. AI 모델 로드 (생략 가능하나 유지)
        model_frame = ttk.LabelFrame(self.frame, text="AI Models")
        model_frame.pack(fill="x", pady=5)
        self.lbl_lstm = ttk.Label(model_frame, text="LSTM: Not Loaded", foreground="gray")
        self.lbl_lstm.pack()
        ttk.Button(model_frame, text="🧠 Load LSTM", command=self.load_lstm).pack(fill="x", padx=5, pady=2)

    def load_map_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            if self.agent.load_map(path):
                self.lbl_map.config(text=os.path.basename(path), foreground="green")
            else:
                messagebox.showerror("에러", "맵 로드 실패")

    def load_lstm(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            self.lbl_lstm.config(text=os.path.basename(path), foreground="blue")

    def adjust_offset(self, dx, dy, reset=False):
        if reset:
            self.map_offset_x = 0
            self.map_offset_y = 0
        else:
            self.map_offset_x += dx
            self.map_offset_y += dy
        self.lbl_offset.config(text=f"Offset: ({self.map_offset_x}, {self.map_offset_y})")