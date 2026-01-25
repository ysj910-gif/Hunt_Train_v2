#ui/roi_selector.py

import tkinter as tk
from tkinter import Toplevel, Canvas, messagebox
from PIL import Image, ImageTk
import cv2

class ROISelector:
    """
    화면의 특정 영역(ROI)을 마우스 드래그로 선택하는 팝업 클래스입니다.
    - 킬 카운트 영역
    - 미니맵 영역
    - 스킬 아이콘 영역 (쿨타임 감지용)
    """
    def __init__(self, parent, agent, target_type, skill_name=None):
        self.parent = parent
        self.agent = agent
        self.target_type = target_type      # 'kill', 'minimap', 'skill'
        self.skill_name = skill_name        # 스킬 아이콘 설정 시 필요
        
        # 1. 화면 캡처 확인
        if not self.agent.vision.window_found:
            messagebox.showwarning("경고", "먼저 메이플스토리 창을 찾아주세요.")
            return

        self.current_frame = self.agent.vision.capture()
        if self.current_frame is None:
            messagebox.showerror("에러", "화면 캡처에 실패했습니다.")
            return
            
        # 2. 팝업창 생성
        self.win = Toplevel(self.parent)
        self.win.title(f"ROI Selector - {target_type}")
        self.win.attributes('-topmost', True) # 항상 위에 표시
        
        # 3. 이미지 변환 (OpenCV BGR -> Pillow RGB)
        img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        
        # 4. 캔버스 생성 및 이미지 표시
        self.canvas = Canvas(self.win, width=self.img_tk.width(), height=self.img_tk.height(), cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw")
        
        # 5. 마우스 이벤트 바인딩
        self.roi_start = None
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 안내 메시지
        print(f"[{target_type}] 영역을 마우스로 드래그하세요.")

    def on_mouse_down(self, event):
        """클릭 시 시작점 기록"""
        self.roi_start = (event.x, event.y)

    def on_mouse_drag(self, event):
        """드래그 시 빨간색 사각형 미리보기 그리기"""
        if self.roi_start:
            self.canvas.delete("roi_rect") # 이전 사각형 지우기
            self.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1], event.x, event.y,
                outline="red", width=2, tag="roi_rect"
            )

    def on_mouse_up(self, event):
        """마우스를 뗐을 때 최종 영역 확정 및 저장"""
        if not self.roi_start:
            return
            
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        
        # 좌표 정렬 (왼쪽 위, 오른쪽 아래)
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)
        
        w = right - left
        h = bottom - top
        
        # 너무 작은 영역은 무시 (오클릭 방지)
        if w < 5 or h < 5:
            return

        rect = (left, top, w, h)
        
        # Vision System에 등록
        try:
            if self.target_type == "kill":
                self.agent.vision.set_roi(rect)
                messagebox.showinfo("설정 완료", f"킬 카운트 영역이 설정되었습니다.\n{rect}")
                
            elif self.target_type == "minimap":
                self.agent.vision.set_minimap_roi(rect)
                messagebox.showinfo("설정 완료", f"미니맵 영역이 설정되었습니다.\n{rect}")
                
            elif self.target_type == "skill":
                if self.skill_name:
                    # 스킬 아이콘 등록 (현재 프레임 정보를 넘겨서 자동 밝기 계산)
                    self.agent.vision.set_skill_roi(self.skill_name, rect, frame=self.current_frame)
                    messagebox.showinfo("설정 완료", f"[{self.skill_name}] 스킬 아이콘이 설정되었습니다.\n(자동 쿨타임 임계값 적용)")
                else:
                    messagebox.showwarning("오류", "스킬 이름이 지정되지 않았습니다.")
                    
            self.win.destroy()
            
        except Exception as e:
            messagebox.showerror("오류", f"영역 설정 중 문제가 발생했습니다: {e}")
            self.win.destroy()