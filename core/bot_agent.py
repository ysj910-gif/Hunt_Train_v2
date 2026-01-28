# core/bot_agent.py

import time
import threading
import traceback
import datetime
import config 
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from utils.logger import logger
from modules.job_manager import JobManager
from engine.skill_strategy import SkillStrategy


# 상태 열거형 (DecisionMaker와 공유)
try:
    from core.decision_maker import BotState
except ImportError:
    from enum import Enum, auto
    class BotState(Enum):
        IDLE = auto()
        MAPS = auto()
        COMBAT = auto()
        EMERGENCY = auto()

if TYPE_CHECKING:
    from modules.vision_system import VisionSystem
    from modules.scanner import GameScanner
    from engine.map_processor import MapProcessor
    from engine.path_finder import PathFinder
    from core.action_handler import ActionHandler
    from core.decision_maker import DecisionMaker
    from core.data_recorder import DataRecorder


class BotAgent:
    def __init__(
        self, 
        vision: "VisionSystem",
        scanner: "GameScanner",
        action_handler: "ActionHandler",
        map_processor: "MapProcessor",
        path_finder: "PathFinder",
        recorder: Optional["DataRecorder"] = None  # Recorder 주입
    ):
        # 1. 의존성 주입
        self.vision = vision
        self.scanner = scanner
        self.action_handler = action_handler
        self.map_processor = map_processor
        self.path_finder = path_finder
        self.recorder = recorder

        self.job_manager = JobManager() 
        self.key_mapping = self.job_manager.get_key_mapping() # jobs.json 값 적용
        logger.info(f"🎹 Key Mapping Initialized: {self.key_mapping}")
        
        self.skill_strategy = SkillStrategy(self.path_finder)
        logger.info("⚔️ Skill Strategy Module Initialized.")

        self.brain: Optional["DecisionMaker"] = None

        # 2. State & Data
        self.state = BotState.IDLE 
        self.running = False
        self.thread = None
        
        self.current_frame = None
        self.player_pos: Optional[Tuple[int, int]] = None
        
        # UI & Debug Data
        self.last_action = "None"
        self.last_action_desc = ""
        self.fps = 0.0
        self.last_loop_time = time.time()
        self.is_recording = False

        logger.info("✅ BotAgent Initialized with injected dependencies.")

    def set_brain(self, brain: "DecisionMaker"):
        self.brain = brain
        logger.info("🧠 Brain (DecisionMaker) attached to Agent.")

    # [복구] UI 호환성을 위한 래퍼 메서드
    def load_map(self, file_path: str) -> bool:
        if self.map_processor.load_map(file_path):
            logger.info(f"Map loaded successfully: {file_path}")
            return True
        else:
            logger.error(f"Failed to load map: {file_path}")
            return False

    # [복구] UI 호환성을 위한 래퍼 메서드
    def set_map_offset(self, x: int, y: int):
        self.map_processor.set_offset(x, y)

    def start(self):
        if self.running: return
        
        if not self.brain:
            logger.critical("⛔ Cannot start Agent: Brain is missing!")
            return

        if not self.map_processor.platforms:
            logger.warning("⚠️ Warning: No map data loaded.")
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        logger.info("🚀 BotAgent Started.")

    def stop(self):
        self.running = False
        self.action_handler.emergency_stop()
        if self.is_recording:
            self.toggle_recording() # 녹화 중지
        if self.thread:
            self.thread.join(timeout=config.THREAD_TIMEOUT)
        logger.info("🛑 BotAgent Stopped.")

    def toggle_recording(self):
        """
        DataRecorder 인스턴스를 사용하여 녹화를 시작/중지합니다.
        전제조건: DataRecorder 클래스에 start_recording(filename)과 stop_recording() 메서드가 구현되어야 함.
        (혹은 open/close)
        """
        if not self.recorder:
            logger.warning("⚠️ DataRecorder module is not injected. Cannot record.")
            return

        if self.is_recording:
            # 기존: self.recorder.close()
            # 메서드 이름은 실제 DataRecorder 구현에 맞춰 수정 필요
            if hasattr(self.recorder, 'close'):
                self.recorder.close()
            elif hasattr(self.recorder, 'stop_recording'):
                self.recorder.stop_recording()
                
            self.is_recording = False
            logger.info("데이터 녹화 종료")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"Record_{timestamp}"
            
            # 기존: self.recorder = DataRecorder(filename)
            # 변경: 주입받은 인스턴스 재사용
            if hasattr(self.recorder, 'open'):
                self.recorder.open(filename)
            elif hasattr(self.recorder, 'start_recording'):
                self.recorder.start_recording(filename)
            else:
                logger.error("DataRecorder에 'open' 또는 'start_recording' 메서드가 없습니다.")
                return

            self.is_recording = True
            logger.info(f"데이터 녹화 시작: {filename}")

    def _main_loop(self):
        self.last_loop_time = time.time()
        
        while self.running:
            try:
                # FPS Calculation
                current_time = time.time()
                delta = current_time - self.last_loop_time
                if delta > 0:
                    weight = config.FPS_SMOOTHING
                    self.fps = self.fps * weight + (1.0 / delta) * (1.0 - weight)
                self.last_loop_time = current_time
                loop_start = time.time()

                # 1. Perception
                self._update_perception()

                # 2. Safety Check
                if not self.vision.window_found:
                    if int(loop_start) % config.WARNING_INTERVAL == 0: 
                        logger.warning("Game window lost...")
                    time.sleep(1.0)
                    continue

                # 3. Decision & Action
                if self.brain:
                    self.brain.update()

                # 4. Recording
                if self.is_recording and self.recorder:
                    # log_step 호출
                    self.recorder.log_step(
                        frame=self.current_frame,
                        player_pos=self.player_pos,
                        action=self.last_action,
                        state=self.state.name if self.state else "UNKNOWN",
                        skill_status=self.scanner.skill_status
                    )

                # Pace Control
                elapsed = time.time() - loop_start
                if elapsed < config.LOOP_INTERVAL:
                    time.sleep(config.LOOP_INTERVAL - elapsed)

            except Exception as e:
                logger.critical(f"Fatal Error in Agent Loop: {e}")
                traceback.print_exc()
                self.state = BotState.EMERGENCY
                # 에러 발생 시 루프를 탈출할지, 계속할지 결정 (여기선 계속 시도)
        
        logger.debug(">>> Agent Loop Terminated")

    def _update_perception(self):
        self.current_frame = self.vision.capture()
        if self.current_frame is not None:
            if self.vision.minimap_roi:
                self.scanner.set_rois(self.vision.minimap_roi, self.vision.kill_roi)
            
            self.player_pos = self.scanner.find_player(self.current_frame)
            self.scanner.update_skill_status(self.current_frame)
            self.scanner.read_kill_count(self.current_frame)

    def get_debug_info(self) -> Dict[str, Any]:
        if not self.running:
            self._update_perception()

        current_plat_idx = -1
        if self.player_pos and self.map_processor:
            px, py = self.player_pos
            curr_plat = self.map_processor.find_current_platform(px, py)
            if curr_plat and curr_plat in self.map_processor.platforms:
                current_plat_idx = self.map_processor.platforms.index(curr_plat)

        return {
            "frame": self.current_frame,
            "player_pos": self.player_pos,
            "state": self.state.name if self.state else "UNKNOWN",
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