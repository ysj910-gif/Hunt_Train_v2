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
        """그리기 로직 업데이트 (포탈, 밧줄, 제작 도구 시각화 추가)"""
        frame = debug_info.get("frame")
        if frame is None: return None
        vis_frame = frame.copy()
        
        h, w = vis_frame.shape[:2]
        
        minimap_roi = debug_info.get("minimap_roi")
        mx, my = (minimap_roi[0], minimap_roi[1]) if minimap_roi else (0, 0)
        
        # 공통 오프셋 계산 (맵 오프셋 + 미니맵 원점)
        off_x = map_offset_x + mx
        off_y = map_offset_y + my

        # --- [1] 기존 로드된 맵 데이터 그리기 (Agent Data) ---
        current_plat_idx = debug_info.get("current_plat_idx", -1)
        
        # 1-1. 발판 (기존: 빨강/초록)
        footholds = debug_info.get("footholds", [])
        for i, plat in enumerate(footholds):
            x1 = int(plat['x_start'] + off_x)
            x2 = int(plat['x_end'] + off_x)
            y = int(plat['y'] + off_y)
            
            if not (0 <= y < h): continue

            if i == current_plat_idx:
                color = (0, 255, 0) # Green (현재 밟고 있는 발판)
                thickness = 4
            else:
                color = (0, 0, 150) # Dark Red (기존 발판)
                thickness = 2
            cv2.line(vis_frame, (x1, y), (x2, y), color, thickness)
            # 번호 표시
            cv2.putText(vis_frame, str(i), (x1, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 1-2. 포탈 (기존: 파랑)
        portals = debug_info.get("portals", [])
        for p in portals:
            src = p['src']; dst = p['dst']
            sx, sy = int(src[0] + off_x), int(src[1] + off_y)
            dx, dy = int(dst[0] + off_x), int(dst[1] + off_y)
            cv2.circle(vis_frame, (sx, sy), 5, (255, 0, 0), 2) # Blue Circle
            cv2.arrowedLine(vis_frame, (sx, sy), (dx, dy), (255, 100, 100), 1)

        # 1-3. 밧줄 (기존: 주황)
        ropes = debug_info.get("ropes", [])
        for r in ropes:
            rx = int(r['x'] + off_x)
            y1 = int(r['y_top'] + off_y)
            y2 = int(r['y_bottom'] + off_y)
            cv2.line(vis_frame, (rx, y1), (rx, y2), (0, 165, 255), 2) # Orange

        # --- [2] 맵 제작 도구 데이터 그리기 (Creator Data) ---
        # MainWindow에서 주입해준 creator_data 참조
        creator = debug_info.get("creator_data", {})
        
        # 2-1. 작성 중인 새 발판 (Cyan)
        for plat in creator.get("new_platforms", []):
            x1 = int(plat['x_start'] + off_x)
            x2 = int(plat['x_end'] + off_x)
            y = int(plat['y'] + off_y)
            cv2.line(vis_frame, (x1, y), (x2, y), (255, 255, 0), 2) # Cyan
            cv2.putText(vis_frame, "NEW", (x1, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 2-2. 작성 중인 새 포탈 (Magenta)
        for p in creator.get("new_portals", []):
            src = p['src']; dst = p['dst']
            sx, sy = int(src[0] + off_x), int(src[1] + off_y)
            dx, dy = int(dst[0] + off_x), int(dst[1] + off_y)
            cv2.circle(vis_frame, (sx, sy), 6, (255, 0, 255), 2)
            cv2.line(vis_frame, (sx, sy), (dx, dy), (255, 0, 255), 1)

        # 2-3. 작성 중인 새 밧줄 (Yellow)
        for r in creator.get("new_ropes", []):
            rx = int(r['x'] + off_x)
            y1 = int(r['y_top'] + off_y)
            y2 = int(r['y_bottom'] + off_y)
            cv2.line(vis_frame, (rx, y1), (rx, y2), (0, 255, 255), 2)

        # 2-4. 작성 중인 맵 이동 포탈 (White Circle)
        for mp in creator.get("new_map_portals", []):
            mx_pos = int(mp['x'] + off_x)
            my_pos = int(mp['y'] + off_y)
            cv2.circle(vis_frame, (mx_pos, my_pos), 8, (255, 255, 255), 2)
            cv2.putText(vis_frame, mp.get("to_map", "MAP"), (mx_pos, my_pos - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2-5. 현재 설정된 시작점/종료점 (매우 중요)
        temp_start = creator.get("temp_start")
        temp_end = creator.get("temp_end")
        
        if temp_start:
            tx, ty = int(temp_start[0] + off_x), int(temp_start[1] + off_y)
            # 노란색 X 표시
            cv2.drawMarker(vis_frame, (tx, ty), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.putText(vis_frame, "START", (tx+5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        if temp_end:
            tx, ty = int(temp_end[0] + off_x), int(temp_end[1] + off_y)
            # 빨간색 X 표시
            cv2.drawMarker(vis_frame, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2)
            cv2.putText(vis_frame, "END", (tx+5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        # --- [3] 플레이어 위치 및 HUD ---
        pos = debug_info.get("player_pos")
        if pos and minimap_roi:
            px, py = pos
            screen_x, screen_y = mx + px, my + py
            cv2.circle(vis_frame, (screen_x, screen_y), 6, (0, 255, 0), -1)

        # HUD Text
        state = debug_info.get("state", "UNKNOWN")
        kill_count = debug_info.get("kill_count", 0)
        
        hud_w, hud_h, margin = 300, 130, 20
        x1 = w - hud_w - margin
        y1 = margin
        x2 = w - margin
        y2 = margin + hud_h
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 0), -1) 
        cv2.putText(vis_frame, f"STATE: {state}", (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"KILL : {kill_count}", (x1 + 20, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
        plat_text = f"PLAT : #{current_plat_idx}" if current_plat_idx != -1 else "PLAT : None"
        cv2.putText(vis_frame, plat_text, (x1 + 20, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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