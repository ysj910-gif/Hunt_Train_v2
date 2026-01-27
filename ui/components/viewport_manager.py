# ui/viewport_manager.py
import cv2
import numpy as np

class ViewportManager:
    """
    줌(Zoom)과 패닝(Pan) 상태를 관리하고,
    원본 좌표 <-> 화면 좌표 변환 및 이미지 리사이징을 담당하는 클래스
    """
    def __init__(self):
        self.zoom_scale = 1.0
        self.pan_x = 0  # 뷰의 중심 X (원본 좌표계)
        self.pan_y = 0  # 뷰의 중심 Y (원본 좌표계)
        self.world_w = 0  # 원본공간 너비 (이미지 너비 또는 맵 너비)
        self.world_h = 0  # 원본공간 높이
        
        # 줌 제한 설정
        self.min_zoom = 0.5
        self.max_zoom = 10.0

    def set_world_size(self, width, height):
        """원본 공간(이미지/맵)의 크기 설정"""
        self.world_w = width
        self.world_h = height
        # 초기화 시 중앙 정렬
        if self.pan_x == 0 and self.pan_y == 0:
            self.center_view()

    def center_view(self):
        """뷰를 중앙으로 초기화"""
        self.pan_x = self.world_w / 2
        self.pan_y = self.world_h / 2
        self.zoom_scale = 1.0

    def adjust_zoom(self, delta):
        """줌 레벨 조정"""
        new_zoom = self.zoom_scale + delta
        self.zoom_scale = max(self.min_zoom, min(self.max_zoom, new_zoom))

    def pan_move(self, dx_screen, dy_screen, canvas_w, canvas_h):
        """
        마우스 드래그(화면 좌표 변화량)에 따른 뷰 중심 이동
        """
        if self.zoom_scale == 0 or canvas_w == 0 or canvas_h == 0: return

        # 화면상 1픽셀이 원본 공간에서 몇 픽셀인지 계산
        # 줌이 클수록 화면 이동에 대해 원본 이동량은 작아짐
        ratio_w = (self.world_w / self.zoom_scale) / canvas_w
        ratio_h = (self.world_h / self.zoom_scale) / canvas_h
        
        # 뷰 중심 이동 (마우스 이동 반대 방향으로 뷰가 이동해야 함)
        self.pan_x -= dx_screen * ratio_w
        self.pan_y -= dy_screen * ratio_h
        
        # 경계 제한 (옵션)
        self.pan_x = max(0, min(self.world_w, self.pan_x))
        self.pan_y = max(0, min(self.world_h, self.pan_y))

    def get_processed_image(self, source_img, canvas_w, canvas_h):
        """
        현재 줌/팬 상태에 맞춰 이미지를 잘라내고(Crop) 리사이징하여 반환
        """
        if source_img is None or canvas_w <= 0 or canvas_h <= 0:
            return None

        # 1. 현재 줌 레벨에서 보여야 할 원본 영역(ROI) 크기 계산
        view_w = self.world_w / self.zoom_scale
        view_h = self.world_h / self.zoom_scale

        # 2. ROI 좌표 계산 (중심점 기준)
        x1 = int(self.pan_x - view_w / 2)
        y1 = int(self.pan_y - view_h / 2)
        x2 = int(x1 + view_w)
        y2 = int(y1 + view_h)

        # 3. 이미지 경계 처리 (음수 좌표나 범위 초과 방지)
        # 패딩을 채워 넣을 수도 있지만, 여기선 간단히 클램핑하고 검은색 배경 유지
        img_h, img_w = source_img.shape[:2]
        
        # 실제 잘라낼 유효 영역
        crop_x1 = max(0, x1); crop_y1 = max(0, y1)
        crop_x2 = min(img_w, x2); crop_y2 = min(img_h, y2)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cropped = source_img[crop_y1:crop_y2, crop_x1:crop_x2]

        # 4. 캔버스 크기로 리사이징
        return cv2.resize(cropped, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)

    def world_to_screen(self, wx, wy, canvas_w, canvas_h):
        """
        원본 좌표(wx, wy)를 화면 좌표(sx, sy)로 변환
        (시뮬레이션 모드에서 벡터 그래픽 그릴 때 사용)
        """
        if canvas_w == 0 or canvas_h == 0: return 0, 0

        # 보여지는 영역의 크기
        view_w = self.world_w / self.zoom_scale
        view_h = self.world_h / self.zoom_scale
        
        # 보여지는 영역의 좌상단 좌표
        view_x = self.pan_x - view_w / 2
        view_y = self.pan_y - view_h / 2
        
        # 비율 계산 후 변환
        sx = (wx - view_x) * (canvas_w / view_w)
        sy = (wy - view_y) * (canvas_h / view_h)
        
        return int(sx), int(sy)