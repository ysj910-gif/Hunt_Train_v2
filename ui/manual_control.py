# ui/manual_control.py

import tkinter as tk
import cv2
import time
from PIL import Image, ImageTk
from utils.logger import logger

class ManualControlWidget(tk.Toplevel):
    """
    Tkinter 기반의 수동 제어 윈도우
    ActionHandler를 통해 입력을 PC A(Target)로 전달합니다.
    """
    def __init__(self, parent, action_handler):
        super().__init__(parent)
        self.action_handler = action_handler
        
        # 윈도우 설정
        self.title('Manual Control Mode')
        self.geometry("1280x760")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.is_active = False
        self.last_send_time = 0
        self.running = True

        self._init_ui()
        
        # 카메라 설정 (VideoThread 대신 after 메서드 사용)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # 키보드 포커스 설정
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        self.focus_set()

        # 영상 루프 시작
        self.update_video()

    def _init_ui(self):
        # 상단 컨트롤 패널
        control_panel = tk.Frame(self, bg="#333", height=50)
        control_panel.pack(side="top", fill="x")

        self.btn_toggle = tk.Button(control_panel, text="🔴 제어 시작 (OFF)", 
                                    bg="#ff4444", fg="white", font=("Arial", 12, "bold"),
                                    command=self.toggle_control)
        self.btn_toggle.pack(side="left", padx=10, pady=5)
        
        info_label = tk.Label(control_panel, text="마우스/키보드 입력이 전송됩니다.", bg="#333", fg="white")
        info_label.pack(side="left", padx=10)

        # 영상 표시 영역 (Canvas 사용)
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # 마우스 이벤트 연결
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", lambda e: self.on_mouse_click(e, 'left', True))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_mouse_click(e, 'left', False))
        self.canvas.bind("<Button-3>", lambda e: self.on_mouse_click(e, 'right', True))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self.on_mouse_click(e, 'right', False))

    def toggle_control(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.btn_toggle.config(text="🟢 제어 중 (ON)", bg="#00cc00")
            self.focus_force() # 키보드 입력을 받기 위해 포커스 강제
            logger.info("Manual Control Activated")
        else:
            self.btn_toggle.config(text="🔴 제어 시작 (OFF)", bg="#ff4444")
            logger.info("Manual Control Deactivated")

    def update_video(self):
        if not self.running: return

        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            
            # [수정] 캔버스 크기에 맞춰 이미지 강제 리사이즈 (좌표 동기화)
            # 이렇게 해야 화면상의 영상 위치와 마우스 좌표 비율이 1:1로 매칭됩니다.
            c_w = self.canvas.winfo_width()
            c_h = self.canvas.winfo_height()
            
            if c_w > 1 and c_h > 1:
                # 성능을 위해 NEAREST 사용 (화질을 원하면 Image.Resampling.BILINEAR 변경 가능)
                img_pil = img_pil.resize((c_w, c_h), Image.Resampling.NEAREST)

            self.imgtk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)

        self.after(33, self.update_video)

    def on_close(self):
        self.running = False
        # [수정] 메인 프로그램과 자원 공유 충돌 방지
        # 캡처보드 특성상 여기서 release()를 호출하면 메인 루프의 연결도 끊어질 수 있으므로,
        # 명시적 해제 없이 윈도우만 파괴합니다. (필요 시 Python GC나 OS가 정리)
        # if self.cap.isOpened():
        #    self.cap.release()
        self.destroy()

    # --- 입력 처리 로직 ---

    def on_mouse_move(self, event):
        if not self.is_active: return
        if time.time() - self.last_send_time < 0.015: return
        self.last_send_time = time.time()

        # Canvas 좌표 -> 1920x1080 비율 매핑
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        
        if c_w == 0 or c_h == 0: return

        ratio_x = event.x / c_w
        ratio_y = event.y / c_h
        
        target_x = int(ratio_x * 1920)
        target_y = int(ratio_y * 1080)

        self.action_handler.mouse_move(target_x, target_y)

    def on_mouse_click(self, event, btn, pressed):
        if not self.is_active: return
        if pressed:
            self.action_handler.mouse_down(btn)
        else:
            self.action_handler.mouse_up(btn)

    def on_key_press(self, event):
        if not self.is_active: return
        key = self._get_key_name(event)
        if key: self.action_handler.key_down(key)
        # [추가] "break"를 반환하여 Alt 키가 윈도우 메뉴를 호출하는 것(렉 유발)을 막습니다.
        return "break"

    def on_key_release(self, event):
        if not self.is_active: return
        key = self._get_key_name(event)
        if key: self.action_handler.key_up(key)
        return "break"

    def _get_key_name(self, event):
        keysym = event.keysym.lower()
        
        if keysym == "??" or not keysym:
            return None

        # [수정] F키, 특수키, 한영키 매핑 추가
        key_map = {
            # 1. 기본 제어키
            'return': 'enter', 'escape': 'esc', 'space': 'space', 'tab': 'tab',
            'backspace': '\x08', # arduino.py 호환
            
            # 2. 조합키
            'control_l': 'ctrl', 'control_r': 'ctrl',
            'shift_l': 'shift', 'shift_r': 'shift',
            'alt_l': 'alt', 'alt_r': 'alt',
            
            # 3. 방향키 및 이동
            'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
            'home': 'home', 'end': 'end', 'prior': 'pageup', 'next': 'pagedown',
            'insert': 'insert', 'delete': 'delete',

            # 4. 기능키 (F1 ~ F12)
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
            'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
            'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',

            # 5. 특수키 및 한영전환 [요청사항]
            'equal': '=',       # '=' 키
            'minus': '-',       # '-' 키
            'hangul': '\x85',   # 한영키 (arduino.py의 매핑값 0x85 적용)
            
            # (필요시 추가) 숫자패드 등
            'kp_enter': 'enter', 'kp_0': '0', 'kp_1': '1', 
        }

        # 1. 매핑된 키 우선 반환
        if keysym in key_map:
            return key_map[keysym]
        
        # 2. 매핑에 없는 일반 문자 (a, b, 1, [, ] 등)는 event.char 사용
        #    keysym은 'bracketleft'처럼 길게 나오므로, 실제 문자('[' 등)가 있는 경우 그걸 사용
        if event.char and len(event.char) == 1:
            return event.char

        # 3. 그 외는 keysym 그대로 반환
        return keysym