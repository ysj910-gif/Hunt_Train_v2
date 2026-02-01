# modules/vision_system.py

import cv2
import mss
import numpy as np
import ctypes
from ctypes import wintypes
import pygetwindow as gw
import time
import config 
from utils.logger import logger, trace_logic

# DPI 인식 설정 (좌표 밀림 방지)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), 
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

class VisionSystem:
    def __init__(self):
        self.capture_area = {
            "top": 0, "left": 0, 
            "width": config.DEFAULT_RES_W, 
            "height": config.DEFAULT_RES_H
        }
        self.window_found = False
        self.hwnd = None

        # [신규 추가] 캡처보드 관련 변수
        self.use_external_cam = getattr(config, 'USE_CAPTURE_CARD', False) # config에 없으면 False
        self.cap = None

        # 캡처 모드면 카메라 초기화 시도
        if self.use_external_cam:
            self._init_camera_device()
        
        self.minimap_roi = None
        self.kill_roi = None
        self.skill_rois = {}
        self.skill_debug_info = {}

    def _get_client_area(self, hwnd):
        """창의 테두리를 제외한 실제 게임 화면 좌표 계산"""
        rect = RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        
        pt = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        
        return pt.x, pt.y, rect.right, rect.bottom

    def find_window(self):
        """메이플스토리 창 탐색 및 좌표 갱신"""

        # [Hook] 캡처보드 모드일 경우 윈도우 탐색 로직을 건너뜀
        if self.use_external_cam:
            if self.cap and self.cap.isOpened():
                self.window_found = True
                return True
            else:
                # 연결이 끊겼다면 재연결 시도
                logger.warning("⚠️ 캡처보드 신호 없음. 재연결 시도...")
                self._init_camera_device()
                return self.window_found
            
        try:
            windows = gw.getWindowsWithTitle('MapleStory')
            if not windows:
                if self.window_found:
                    logger.warning("메이플스토리 창을 찾을 수 없습니다.")
                self.window_found = False
                return False
            
            win = windows[0]
            self.hwnd = win._hWnd
            
            # 최소화 상태면 복구
            if win.isMinimized:
                logger.info("최소화된 창을 복구합니다.")
                win.restore()
                time.sleep(config.WINDOW_RESTORE_DELAY)            
            # 클라이언트 영역 계산
            rect = self._get_client_area(self.hwnd)
            if not rect:
                return False
            
            x, y, w, h = rect
            if w <= 0 or h <= 0:
                return False

            new_area = {
                "top": int(y), 
                "left": int(x), 
                "width": int(w), 
                "height": int(h)
            }
            
            # 위치가 변했을 때만 로그 출력
            if new_area != self.capture_area:
                self.capture_area = new_area
                logger.info(f"게임 창 위치 갱신: {self.capture_area}")

            self.window_found = True
            return True

        except Exception as e:
            logger.error(f"창 탐색 중 오류 발생: {e}")
            self.window_found = False
            return False

    def capture(self):
        """현재 화면을 캡처하여 OpenCV 포맷으로 반환"""

            
        # [Hook] 캡처보드 모드 처리
        if self.use_external_cam:
            if not self.cap or not self.cap.isOpened():
                logger.error("캡처 장치가 준비되지 않았습니다.")
                self.find_window() # 재연결 시도
                return None
            
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("프레임 수신 실패 (캡처보드)")
                return None
            
            # 필요 시 해상도 리사이징 (봇이 예상하는 해상도와 다를 경우)
            # if frame.shape[1] != config.DEFAULT_RES_W:
            #     frame = cv2.resize(frame, (config.DEFAULT_RES_W, config.DEFAULT_RES_H))
                
            return frame
        
        if not self.window_found or self.capture_area["width"] <= 0:
            if not self.find_window():
                return None

        try:
            # [수정] mss를 캡처 시점에 with문으로 생성하여 스레드 안전성 확보
            with mss.mss() as sct:
                # mss.grab은 모니터 범위를 벗어나면 에러 발생 가능
                img_buffer = sct.grab(self.capture_area)
                img_np = np.array(img_buffer)
                frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
                return frame

        except Exception as e:
            # logger.exception("화면 캡처 실패") # 너무 자주 뜨면 로그 파일 커지므로 주석 처리 권장
            self.window_found = False # 다음 루프 재탐색 유도
            return None
            
    # ==========================================================
    # [신규 기능 4, 5] ROI 설정 및 관리 메서드
    # ==========================================================

    def set_roi(self, rect):
        """킬 카운트 영역 설정 (GUI 호환용)"""
        self.kill_roi = rect
        logger.info(f"킬 카운트 ROI 설정됨: {rect}")

    def set_minimap_roi(self, rect):
        """미니맵 영역 설정"""
        self.minimap_roi = rect
        logger.info(f"미니맵 ROI 설정됨: {rect}")

    def set_skill_roi(self, name, rect, frame=None, threshold=None):
        """
        스킬 아이콘 ROI 및 쿨타임 임계값 설정
        """
        # Threshold 자동 계산: 현재 활성화된 상태라고 가정하고 밝기의 70%를 기준으로 잡음
        if threshold is None and frame is not None:
            x, y, w, h = rect
            # 프레임 범위 체크
            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    v_mean = np.mean(hsv[:, :, 2])
                    threshold = v_mean * config.SKILL_THRESH_RATIO
                    logger.debug(f"[{name}] 자동 임계값 계산: 현재 밝기 {v_mean:.1f} -> 기준 {threshold:.1f}")
            
        if threshold is None: 
            threshold = config.DEFAULT_SKILL_THRESH
        self.skill_rois[name] = {'rect': rect, 'threshold': threshold}
        logger.info(f"스킬 ROI 등록: {name} {rect} (Thresh: {threshold:.1f})")

    # ==========================================================
    # [신규 기능 5] 스킬 쿨타임(채도/명도) 분석 로직
    # ==========================================================

    def check_skill_cooldown(self, frame):
        """
        등록된 모든 스킬 아이콘을 분석하여 쿨타임 여부를 반환합니다.
        """
        status = {}
        self.skill_debug_info = {} # UI 표시용 초기화
        
        for name, data in self.skill_rois.items():
            is_cool, val = self._analyze_single_skill(frame, data)
            status[name] = is_cool
            
            # UI 디버깅을 위한 정보 저장
            self.skill_debug_info[name] = {
                'val': val,
                'thr': data['threshold'],
                'is_cool': is_cool
            }
            
        return status
        
    def is_skill_on_cooldown(self, name, frame):
        """특정 스킬 하나만 확인"""
        if name not in self.skill_rois:
            return False
        is_cool, _ = self._analyze_single_skill(frame, self.skill_rois[name])
        return is_cool

    def _analyze_single_skill(self, frame, data):
        """단일 스킬 영역 분석 (내부 메서드)"""
        x, y, w, h = data['rect']
        thresh = data['threshold']
        
        # 화면 범위 안전장치
        if y+h > frame.shape[0] or x+w > frame.shape[1]:
            return False, 0.0

        roi = frame[y:y+h, x:x+w]
        
        # HSV 변환 후 V(명도) 채널 평균 계산
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        current_val = np.mean(hsv[:, :, 2])
        
        # 현재 밝기가 기준값보다 낮으면(어두우면) 쿨타임 중으로 판단
        is_cooldown = current_val < thresh
        
        return is_cooldown, current_val

    def capture_and_analyze(self):
        """
        [GUI 호환용] 캡처 후 기본적인 분석 정보를 반환
        """
        frame = self.capture()
        if frame is None:
            return None, 0, 0, 0, 0
        
        # 호환성을 위해 GUI에서 사용하는 형태로 반환
        return frame, 0, 0, 0, 0
    
    
    def activate_window(self):
        """게임 창을 맨 앞으로 가져오고 포커스를 줍니다."""
        if not self.hwnd:
            return False
            
        try:
            # 현재 활성화된 창이 이미 메이플이면 패스
            foreground_hwnd = user32.GetForegroundWindow()
            if foreground_hwnd == self.hwnd:
                return True

            # 최소화 상태면 복구
            if user32.IsIconic(self.hwnd):
                user32.ShowWindow(self.hwnd, 9) # SW_RESTORE
            
            # 강제 포커스 (Alt key trick to bypass Windows restriction)
            # 윈도우는 다른 프로그램이 포커스를 뺏어가는 것을 막는 경우가 있어, Alt키를 누르는 척 하면서 전환
            user32.keybd_event(0, 0, 0, 0)
            user32.SetForegroundWindow(self.hwnd)
            
            logger.debug("게임 창 활성화 시도...")
            time.sleep(config.WINDOW_ACTIVATE_DELAY)
            return True
        except Exception as e:
            logger.error(f"창 활성화 실패: {e}")
            return False
        
    def _init_camera_device(self):
        """[신규] 캡처보드/웹캠 장치 초기화"""
        try:
            cam_idx = getattr(config, 'CAMERA_INDEX', 0)
            logger.info(f"🎥 캡처보드 연결 시도 (Index: {cam_idx})...")
            
            # Windows에서는 cv2.CAP_DSHOW가 초기화 속도가 빠름
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error("❌ 캡처보드를 열 수 없습니다. 장치 연결을 확인하세요.")
                self.window_found = False
            else:
                # 해상도 강제 설정 (선택 사항)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DEFAULT_RES_W)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.DEFAULT_RES_H)
                
                # 버퍼 비우기 (딜레이 방지)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                logger.info(f"✅ 캡처보드 연결 성공. 해상도 설정: {config.DEFAULT_RES_W}x{config.DEFAULT_RES_H}")
                self.window_found = True # 카메라는 '창'이 아니지만, 로직 흐름상 Found로 처리

    
        except Exception as e:
            logger.error(f"카메라 초기화 중 예외 발생: {e}")

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()