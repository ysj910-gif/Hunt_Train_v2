# core/bot_agent.py

import time
import threading
import traceback
import datetime
from typing import Tuple, Optional, Dict, Any

import config 

# Modules
from modules.vision_system import VisionSystem
from modules.scanner import GameScanner
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from engine.physics_engine import PhysicsEngine
from core.action_handler import ActionHandler
from core.model_loader import ModelLoader
from utils.logger import logger, trace_logic

# [신규] 리팩토링된 의사결정 모듈 임포트
from core.decision_maker import DecisionMaker, BotState

try:
    from core.data_recorder import DataRecorder
except ImportError:
    DataRecorder = None

# (호환성 유지) SkillStrategy는 DecisionMaker나 PathFinder 내부에서 쓰일 수 있음
try:
    from engine.skill_strategy import SkillStrategy
except ImportError:
    SkillStrategy = None

class BotAgent:
    """
    메이플스토리 봇의 중앙 제어 에이전트.
    모듈 초기화, 메인 루프 실행, 데이터 공급(UI)을 담당합니다.
    복잡한 판단 로직은 self.brain (DecisionMaker)으로 위임되었습니다.
    """

    def __init__(self, map_file_path: str = None):
        # 1. Perception Layer
        self.vision = VisionSystem()
        self.scanner = GameScanner()

        # 2. Intelligence Layer
        self.map_processor = MapProcessor()
        self.physics_engine = PhysicsEngine()
        # 물리 엔진 모델 로드 (없으면 Fallback)
        self.physics_engine.load_model("physics_hybrid_model.pth")
        
        self.path_finder = PathFinder(self.map_processor, self.physics_engine)

        if map_file_path:
            self.load_map(map_file_path)

        # SkillStrategy (Optional)
        self.skill_strategy = SkillStrategy(self.path_finder) if SkillStrategy else None
        
        self.model_loader = ModelLoader()

        # 3. Control Layer (ActionHandler)
        mode = "SOFTWARE"
        port = getattr(config, 'SERIAL_PORT', None)
        if port:
            logger.info(f"아두이노 포트 감지됨: {port} -> 하드웨어 모드 활성화")
            mode = "HARDWARE"
        self.action_handler = ActionHandler(mode=mode, serial_port=port)

        # 4. [신규] Brain (Decision Maker)
        # Agent 자신(self)을 넘겨주어 Brain이 모든 모듈에 접근할 수 있게 함
        self.brain = DecisionMaker(self)

        # 5. State & Data
        # 상태값은 Brain이 주로 변경하지만, UI와의 연동을 위해 Agent에도 속성을 유지합니다.
        self.state = BotState.IDLE 
        self.running = False
        self.thread = None
        self.current_frame = None
        self.player_pos: Optional[Tuple[int, int]] = None
        
        # UI 표시용 변수
        self.last_action = "None"
        self.last_action_desc = ""
        self.fps = 0.0
        self.last_loop_time = time.time()
        
        self.key_mapping = {}
        self.recorder = None
        self.is_recording = False

        logger.info(f"BotAgent Initialized ({mode} Mode).")

    def load_map(self, file_path: str) -> bool:
        if self.map_processor.load_map(file_path):
            logger.info(f"Map loaded successfully: {file_path}")
            return True
        else:
            logger.error(f"Failed to load map: {file_path}")
            return False
        
    def set_map_offset(self, x: int, y: int):
        self.map_processor.set_offset(x, y)

    def start(self):
        if self.running: return
        if not self.map_processor.platforms:
            logger.warning("주의: 로드된 맵 데이터가 없습니다.")
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        logger.info("BotAgent Started.")

    def stop(self):
        self.running = False
        self.action_handler.emergency_stop()
        if self.is_recording:
            self.toggle_recording()
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("BotAgent Stopped.")

    def toggle_recording(self):
        if self.is_recording:
            if self.recorder:
                self.recorder.close()
                self.recorder = None
            self.is_recording = False
            logger.info("데이터 녹화 종료")
        else:
            if DataRecorder:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"Record_{timestamp}"
                self.recorder = DataRecorder(filename)
                self.is_recording = True
                logger.info(f"데이터 녹화 시작: {filename}")
            else:
                logger.error("DataRecorder 모듈 없음")

    def set_state(self, state_name: str):
        """GUI에서 상태 변경 요청 시 호출"""
        try:
            # DecisionMaker에 정의된 BotState 사용
            new_state = BotState[state_name.upper()]
            self.state = new_state
            logger.info(f"State changed manually to: {self.state}")
        except KeyError:
            logger.error(f"Invalid state name: {state_name}")

    @trace_logic
    def _main_loop(self):
        self.last_loop_time = time.time()
        print(">>> [DEBUG] _main_loop 진입")

        while self.running:
            try:
                # FPS 계산
                current_time = time.time()
                delta = current_time - self.last_loop_time
                if delta > 0:
                    self.fps = self.fps * 0.9 + (1.0 / delta) * 0.1
                self.last_loop_time = current_time
                loop_start = time.time()

                # 1. Perception Update
                self._update_perception()

                # 2. Safety Check
                if not self.vision.window_found:
                    if int(loop_start) % 5 == 0: 
                        logger.warning("Game window not found...")
                    time.sleep(1.0)
                    continue

                # 3. Decision & Action (Brain에게 위임)
                # Brain이 self.state를 읽고 적절한 행동(_handle_combat 등)을 수행함
                self.brain.update()

                # 4. Data Recording
                if self.is_recording and self.recorder:
                    self.recorder.log_step(
                        frame=self.current_frame,
                        player_pos=self.player_pos,
                        action=self.last_action,
                        state=self.state.name,
                        skill_status=self.scanner.skill_status
                    )

                # Loop Pace Control
                elapsed = time.time() - loop_start
                if elapsed < 0.05:
                    time.sleep(0.05 - elapsed)

            except Exception as e:
                print(f">>> [CRITICAL ERROR] {e}")
                traceback.print_exc()
                logger.error(f"Critical Error: {e}")
                self.state = BotState.EMERGENCY
        
        print(">>> [DEBUG] _main_loop 종료")

    def _update_perception(self):
        """화면 인식 및 데이터 갱신"""
        self.current_frame = self.vision.capture()
        if self.current_frame is not None:
            # ROI 동기화
            if self.vision.minimap_roi:
                self.scanner.set_rois(self.vision.minimap_roi, self.vision.kill_roi)

            # 플레이어 위치 및 스킬 상태 갱신
            self.player_pos = self.scanner.find_player(self.current_frame)
            self.scanner.update_skill_status(self.current_frame)
            self.scanner.read_kill_count(self.current_frame)

    def get_debug_info(self) -> Dict[str, Any]:
        """UI에 표시할 데이터 반환"""
        if not self.running:
            self._update_perception()

        current_plat_idx = -1
        if self.player_pos:
            px, py = self.player_pos
            curr_plat = self.map_processor.find_current_platform(px, py)
            if curr_plat and curr_plat in self.map_processor.platforms:
                current_plat_idx = self.map_processor.platforms.index(curr_plat)

        info = {
            "frame": self.current_frame,
            "player_pos": self.player_pos,
            "state": self.state.name,
            "action": self.last_action,
            "action_desc": self.last_action_desc,
            "fps": self.fps,
            "kill_count": self.scanner.current_kill_count,
            "current_plat_idx": current_plat_idx,
            "footholds": self.map_processor.platforms,
            "minimap_roi": self.vision.minimap_roi,
            "kill_roi": self.vision.kill_roi,
            "install_skills": getattr(self.path_finder, 'installed_objects', {}),
            "skill_debug": getattr(self.vision, 'skill_debug_info', {}) 
        }
        return info