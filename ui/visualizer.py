# ui/visualizer.py
import cv2
import numpy as np
from PIL import Image, ImageTk

class Visualizer:
    @staticmethod
    def resize_image_keep_ratio(image, target_w, target_h):
        """이미지의 비율을 유지하면서 target 사이즈 안에 맞게 리사이징 (Letterbox 방식)"""
        h, w = image.shape[:2]
        
        if target_w < 1 or target_h < 1:
            return image, 1.0, (0, 0)

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, scale, (x_offset, y_offset)

    @staticmethod
    def draw_debug_view(debug_info, map_offset_x, map_offset_y):
        """그리기 로직 (HUD 위치 우측 상단으로 변경)"""
        frame = debug_info.get("frame")
        if frame is None: return None
        vis_frame = frame.copy()
        
        # 화면 크기 가져오기 (위치 계산용)
        h, w = vis_frame.shape[:2]
        
        minimap_roi = debug_info.get("minimap_roi")
        mx, my = (minimap_roi[0], minimap_roi[1]) if minimap_roi else (0, 0)
        
        # 데이터 가져오기
        current_plat_idx = debug_info.get("current_plat_idx", -1)
        kill_count = debug_info.get("kill_count", 0)

        # 1. 발판 그리기
        footholds = debug_info.get("footholds", [])
        for i, plat in enumerate(footholds):
            x1 = plat['x_start'] + map_offset_x + mx
            x2 = plat['x_end'] + map_offset_x + mx
            y = plat['y'] + map_offset_y + my
            
            if not (0 <= y < vis_frame.shape[0]):
                continue

            if i == current_plat_idx:
                color = (0, 255, 0) # Green (현재 발판)
                thickness = 4
            else:
                color = (0, 0, 255) # Red
                thickness = 2
                
            cv2.line(vis_frame, (int(x1), int(y)), (int(x2), int(y)), color, thickness)
            
            # 발판 번호 표시
            cv2.putText(vis_frame, str(i), (int(x1), int(y) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 2. 캐릭터 위치
        pos = debug_info.get("player_pos")
        if pos and minimap_roi:
            px, py = pos
            screen_x, screen_y = mx + px, my + py
            cv2.circle(vis_frame, (screen_x, screen_y), 6, (0, 255, 0), -1)

        # 3. HUD 텍스트 (위치 변경: 좌측 -> 우측 상단)
        state = debug_info.get("state", "UNKNOWN")
        
        # HUD 박스 크기 설정
        hud_w = 300
        hud_h = 130
        margin = 20
        
        # 우측 상단 좌표 계산 (화면 너비 - HUD너비 - 여백)
        x1 = w - hud_w - margin
        y1 = margin
        x2 = w - margin
        y2 = margin + hud_h
        
        # 배경 박스 그리기
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 0), -1) 
        
        # 텍스트 그리기 (x1 좌표 기준으로 배치)
        cv2.putText(vis_frame, f"STATE: {state}", (x1 + 20, y1 + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(vis_frame, f"KILL : {kill_count}", (x1 + 20, y1 + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
        
        plat_text = f"PLAT : #{current_plat_idx}" if current_plat_idx != -1 else "PLAT : None"
        cv2.putText(vis_frame, plat_text, (x1 + 20, y1 + 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return vis_frame

    @staticmethod
    def convert_to_tk_image(cv_image, target_w=None, target_h=None):
        """Tkinter 이미지 변환 (리사이징 옵션 추가)"""
        if cv_image is None: return None
        
        final_img = cv_image
        if target_w and target_h:
            final_img, _, _ = Visualizer.resize_image_keep_ratio(cv_image, target_w, target_h)
            
        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(image=img_pil)