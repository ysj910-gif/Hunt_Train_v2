# core/bot_agent.py

import time
import threading
import traceback
import datetime
from enum import Enum, auto
from typing import Tuple, Optional, Dict, Any

# [추가] 설정 파일 임포트
import config 

# Modules & Engine
from modules.vision_system import VisionSystem
from modules.scanner import GameScanner
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from engine.physics_engine import PhysicsEngine
from core.action_handler import ActionHandler
from core.model_loader import ModelLoader
from utils.logger import logger, trace_logic


# [신규] DataRecorder (추후 구현될 파일, 없을 경우를 대비해 예외 처리)
try:
    from core.data_recorder import DataRecorder
except ImportError:
    DataRecorder = None
    # logger.warning("DataRecorder module not found. Recording feature will be disabled.")

# (옵션) SkillStrategy
try:
    from engine.skill_strategy import SkillStrategy
except ImportError:
    SkillStrategy = None
    logger.warning("SkillStrategy module not found. Combat logic will be limited.")

class BotState(Enum):
    IDLE = auto()       # 대기 상태
    MAPS = auto()       # 맵 이동 및 순찰
    COMBAT = auto()     # 전투 상태
    EMERGENCY = auto()  # 에러 발생 및 복구

class BotAgent:
    """
    메이플스토리 봇의 중앙 제어 에이전트.
    Perception(Vision) -> Intelligence(PathFinder) -> Control(Action) 파이프라인을 조율합니다.
    UI에 데이터를 공급하고(Data Provider), 녹화 기능을 제어합니다.
    """

    def __init__(self, map_file_path: str = None):
        # 1. Perception Layer
        self.vision = VisionSystem()
        self.scanner = GameScanner()

        # 2. Intelligence Layer
        self.map_processor = MapProcessor()
        
        # [수정] Physics Engine 로드 및 PathFinder 연결
        self.physics_engine = PhysicsEngine()
        # 모델 경로가 있다면 로드, 없으면 기본값 (나중에 UI에서 로드 가능)
        self.physics_engine.load_model("physics_hybrid_model.pth") 
        
        self.path_finder = PathFinder(self.map_processor, self.physics_engine)

        if map_file_path:
            self.load_map(map_file_path)

        if SkillStrategy:
            self.skill_strategy = SkillStrategy(self.path_finder)
        else:
            self.skill_strategy = None

        self.model_loader = ModelLoader()

        # [★ 핵심 수정] ActionHandler를 하드웨어 모드로 초기화
        # config.SERIAL_PORT가 설정되어 있다면 HARDWARE 모드로 시도합니다.
        mode = "SOFTWARE"
        port = getattr(config, 'SERIAL_PORT', None)
        
        if port:
            logger.info(f"아두이노 포트 감지됨: {port} -> 하드웨어 모드 활성화")
            mode = "HARDWARE"
        
        self.action_handler = ActionHandler(mode=mode, serial_port=port)

        # 4. Data Recording
        self.recorder = None
        self.is_recording = False

        # ... (나머지 초기화 코드 기존과 동일) ...
        self.state = BotState.IDLE
        self.running = False
        self.thread = None
        self.current_frame = None
        self.player_pos: Optional[Tuple[int, int]] = None
        self.last_action = "None"
        self.last_action_desc = ""
        self.fps = 0.0
        self.last_loop_time = time.time()
        
        # 키 매핑 저장소 (UI에서 주입됨)
        self.key_mapping = {}

        logger.info(f"BotAgent Initialized ({mode} Mode).")

    def load_map(self, file_path: str) -> bool:
        """JSON 맵 파일을 로드합니다."""
        if self.map_processor.load_map(file_path):
            logger.info(f"Map loaded successfully: {file_path}")
            return True
        else:
            logger.error(f"Failed to load map: {file_path}")
            return False
        
    def set_map_offset(self, x: int, y: int):
        """UI에서 변경된 오프셋을 MapProcessor에 실시간으로 반영합니다."""
        if self.map_processor:
            self.map_processor.set_offset(x, y)
            # logger.info(f"Map offset updated to: ({x}, {y})") # 필요 시 주석 해제

    def start(self):
        """봇 메인 루프 시작"""
        if self.running:
            return
        
        if not self.map_processor.platforms:
            logger.warning("주의: 로드된 맵 데이터가 없습니다. 봇이 정상 작동하지 않을 수 있습니다.")
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        logger.info("BotAgent Started.")

    def stop(self):
        """봇 중지"""
        self.running = False
        self.action_handler.emergency_stop()
        
        # 녹화 중이었다면 저장하고 종료
        if self.is_recording:
            self.toggle_recording()
            
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("BotAgent Stopped.")

    # =========================================================================
    # [신규 기능 2, 6] UI 데이터 공급 메서드
    # =========================================================================
    #@trace_logic
    def get_debug_info(self) -> Dict[str, Any]:
        """UI 시각화를 위한 정보 패키징"""
        if not self.running:
            self._update_perception()

        # 현재 밟고 있는 발판 찾기 및 인덱스 추출
        current_plat_idx = -1
        if self.player_pos:
            px, py = self.player_pos
            # map_processor를 통해 현재 발판 객체를 찾음
            curr_plat = self.map_processor.find_current_platform(px, py)
            if curr_plat and curr_plat in self.map_processor.platforms:
                current_plat_idx = self.map_processor.platforms.index(curr_plat)

        info = {
            "frame": self.current_frame,
            "player_pos": self.player_pos,
            "state": self.state.name,
            "action": self.last_action,
            "action_desc": self.last_action_desc,

            "fps": self.fps, # <--- UI로 FPS 전달
            
            # [추가] 킬 카운트 & 현재 발판 인덱스 전달
            "kill_count": self.scanner.current_kill_count,
            "current_plat_idx": current_plat_idx,

            "footholds": self.map_processor.platforms,
            "minimap_roi": self.vision.minimap_roi,
            "kill_roi": self.vision.kill_roi,
            "install_skills": getattr(self.path_finder, 'installed_objects', {}),
            "skill_debug": getattr(self.vision, 'skill_debug_info', {}) 
        }
        return info
    # =========================================================================
    # [신규 기능 9] 데이터 녹화 제어 메서드
    # =========================================================================
    def toggle_recording(self):
        """데이터 녹화 모듈을 켜거나 끕니다."""
        if self.is_recording:
            # 녹화 종료
            if self.recorder:
                self.recorder.close()
                self.recorder = None
            self.is_recording = False
            logger.info("데이터 녹화가 종료되었습니다.")
        else:
            # 녹화 시작
            if DataRecorder:
                # 파일명 생성 (예: Record_20231025_1230.csv)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"Record_{timestamp}"
                
                self.recorder = DataRecorder(filename)
                self.is_recording = True
                logger.info(f"데이터 녹화가 시작되었습니다. (파일명: {filename})")
            else:
                logger.error("DataRecorder 모듈이 없어 녹화를 시작할 수 없습니다.")

    @trace_logic
    def _main_loop(self):
        """
        Main Control Loop:
        Vision -> Scanner -> Intelligence -> Action
        """
        self.last_loop_time = time.time()
        
        # [수정 1] 루프 진입 확인용 로그 추가
        print(">>> [DEBUG] _main_loop 진입 성공! (스레드 시작됨)")

        while self.running:
            try:
                current_time = time.time()
                delta = current_time - self.last_loop_time
                if delta > 0:
                    current_fps = 1.0 / delta
                    # 이전 값 90%, 새 값 10% 반영하여 부드럽게 표시
                    self.fps = self.fps * 0.9 + current_fps * 0.1
                self.last_loop_time = current_time

                loop_start = time.time()

                # --- 1. Perception Phase ---
                # [수정 2] 인식 단계 전후로 로그 추가 (화면 캡처 등에서 멈추는지 확인)
                # print(">>> [DEBUG] 1. 인식 시작") 
                self._update_perception()
                # print(">>> [DEBUG] 1. 인식 완료")

                # --- 2. Safety Check ---
                if not self.vision.window_found:
                    # 너무 잦은 로그 방지
                    if int(loop_start) % 5 == 0: 
                        logger.warning("Game window not found. Waiting...")
                    time.sleep(1.0)
                    continue

                if self.player_pos is None and self.state != BotState.IDLE:
                    # logger.warning("Player position lost during active state.")
                    time.sleep(0.1)
                    continue

                # --- 3. FSM Decision Phase ---
                # 상태별 핸들러 실행
                if self.state == BotState.IDLE:
                    self._handle_idle()
                
                elif self.state == BotState.MAPS:
                    self._handle_maps()
                
                elif self.state == BotState.COMBAT:
                    self._handle_combat()
                
                elif self.state == BotState.EMERGENCY:
                    self._handle_emergency()

                # --- 4. Data Recording (Loop End) ---
                if self.is_recording and self.recorder:
                    # 현재 프레임의 데이터를 레코더에 전달
                    self.recorder.log_step(
                        frame=self.current_frame,
                        player_pos=self.player_pos,
                        action=self.last_action,
                        state=self.state.name,
                        skill_status=self.scanner.skill_status
                    )

                # Loop Pace Control (Max 20 FPS logic updates)
                elapsed = time.time() - loop_start
                if elapsed < 0.05:
                    time.sleep(0.05 - elapsed)

            except Exception as e:
                # [수정 3] 에러 발생 시 콘솔에 즉시 출력하도록 변경
                print(f">>> [CRITICAL THREAD ERROR] {e}")
                import traceback
                traceback.print_exc()

                logger.error(f"Critical Error in Main Loop: {e}")
                logger.error(traceback.format_exc())
                self.state = BotState.EMERGENCY
        
        # [수정 4] 루프 종료 로그
        print(">>> [DEBUG] _main_loop 종료됨 (Running=False)")

    #@trace_logic
    def _update_perception(self):
        """화면 캡처 및 상태 갱신"""
        self.current_frame = self.vision.capture()
        if self.current_frame is not None:
            
            # [기존 코드] VisionSystem의 ROI 정보를 Scanner에 동기화
            if self.vision.minimap_roi:
                self.scanner.set_rois(self.vision.minimap_roi, self.vision.kill_roi)

            # [기존 코드] 스캐너 업데이트
            self.player_pos = self.scanner.find_player(self.current_frame)
            self.scanner.update_skill_status(self.current_frame)

            # [▼ 추가된 코드] 좌표 기반 발판 추론 과정 기록
            if self.player_pos:
                px, py = self.player_pos
                # 1. 맵 프로세서에게 "나 지금 어느 발판 위에 있어?"라고 물어봄
                found_plat = self.map_processor.find_current_platform(px, py)
                
                # 2. 발판 ID 추출 (리스트 내 인덱스)
                plat_id = -1
                if found_plat in self.map_processor.platforms:
                    plat_id = self.map_processor.platforms.index(found_plat)

                # 3. 의사결정 로그에 기록 (decision_history.jsonl에 저장됨)
                # -> 나중에 "좌표는 (100, 200)인데 왜 발판은 None이지?"를 분석 가능
                logger.log_decision(
                    step="Perception",          # 단계
                    state=self.state.name,      # 현재 봇 상태
                    decision=f"On_Plat_{plat_id}", # 결론: "발판 X번 위에 있음"
                    reason="Position Updated",  # 이유
                    current_pos=self.player_pos, # 입력 데이터 (좌표)
                    platform_y=found_plat['y'] if found_plat else "N/A" # 인식된 발판 높이
                )

    # --- State Handlers ---

    @trace_logic
    def _handle_idle(self):
        """대기 상태"""
        self.last_action = "Idle"
        self.last_action_desc = "Waiting for user command..."
        time.sleep(0.1)

    @trace_logic
    def _handle_maps(self):
        """맵 순찰 및 이동"""
        if not self.player_pos:
            return

        target_pos = self.path_finder.find_next_patrol_target(self.player_pos)
        
        if not target_pos:
            logger.log_decision("BotAgent", "MAPS", "Switch to COMBAT", "No waypoint")
            self.state = BotState.COMBAT
            return

        target_x, target_y = target_pos
        
        # 상태 업데이트 (UI 표시용)
        self.last_action = "Moving"
        self.last_action_desc = f"To ({target_x}, {target_y})"

        # Action 실행
        success = self.action_handler.move_x(
            target_x, 
            get_current_pos=lambda: self.scanner.find_player(self.vision.capture())
        )

        if not success:
            logger.warning("Movement failed. Determining recovery action...")
            self.state = BotState.EMERGENCY

    @trace_logic
    def _handle_combat(self):
        """전투 로직 (키 매핑 적용 수정)"""

        if int(time.time()) % 10 == 0:
            self.vision.activate_window()
        
        if not self.player_pos:
            return

        action_plan = None
        skill_key = None 

        # 1. 스킬 전략 모듈에게 어떤 '스킬 이름'을 쓸지 물어봄
        if self.skill_strategy:
             action_plan = self.skill_strategy.decide_skill(self.player_pos, self.scanner.skill_status)

        # 키 매핑 조회 로직
        jump_key = 'alt'
        attack_key = 'ctrl'
        
        if hasattr(self, 'key_mapping'):
            if action_plan and action_plan in self.key_mapping:
                skill_key = self.key_mapping[action_plan]
            
            if 'jump' in self.key_mapping: jump_key = self.key_mapping['jump']
            if 'attack' in self.key_mapping: attack_key = self.key_mapping['attack']

        if not action_plan:
            action_plan = "basic_attack"

        # 상태 업데이트 (UI 표시용)
        self.last_action = action_plan
        self.last_action_desc = "Combat Routine"

        # 3. Action 실행
        if action_plan == "basic_attack":
             # TODO: 몬스터 위치 기반 방향 결정 로직 (scanner에서 몬스터 정보를 가져온다고 가정)
             # 예: monsters = self.scanner.get_monsters() ...
             
             direction = 'left' # 현재는 하드코딩 되어 있음
             decision_reason = "Default direction (No monster logic)" # 이유 설명

             # [추가] 여기에 의사결정 로그를 남깁니다.
             logger.log_decision(
                 step="BotAgent",
                 state="COMBAT",
                 decision=f"JumpShot_{direction.upper()}",
                 reason=decision_reason,
                 details={"skill": "Basic Attack", "keys": f"{jump_key}+{attack_key}"}
             )

             # [수정] 찾아낸 점프/공격 키를 사용하여 점프샷
             self.action_handler.jump_shot(direction, jump_key=jump_key, attack_key=attack_key)
        else:
            # 스킬 사용의 경우 SkillStrategy 내부에서 이미 로그를 남기지만,
            # 실행 시점에도 기록하고 싶다면 아래 주석을 해제하세요.
            # logger.log_decision("BotAgent", "COMBAT", f"Cast_{action_plan}", "Skill Strategy Executed")
            
            # 결정된 스킬 키가 있으면 누름
            if skill_key:
                self.action_handler.press(skill_key)
            else:
                logger.warning(f"스킬 [{action_plan}]에 매핑된 키를 찾을 수 없습니다.")

    def _handle_emergency(self):
        """에러 복구"""
        self.last_action = "Recovering"
        self.last_action_desc = "Emergency Mode"
        
        logger.warning("Entering EMERGENCY Recovery Mode.")
        self.action_handler.emergency_stop()
        time.sleep(1.0)

        logger.log_decision("BotAgent", "EMERGENCY", "Attempt Recovery", "Random Jump")
        self.action_handler.press('jump')
        time.sleep(0.5)
        
        self.current_frame = self.vision.capture()
        new_pos = self.scanner.find_player(self.current_frame)
        
        if new_pos:
            logger.info("Position recovered. Returning to IDLE.")
            self.state = BotState.IDLE 
        else:
            logger.critical("Recovery failed. Stopping Bot.")
            self.stop()

    def set_state(self, state_name: str):
        """외부(GUI)에서 상태를 변경하기 위한 메서드"""
        try:
            new_state = BotState[state_name.upper()]
            self.state = new_state
            logger.info(f"State changed manually to: {self.state}")
        except KeyError:
            logger.error(f"Invalid state name: {state_name}")

    def _update_perception(self):
        """화면 캡처 및 상태 갱신"""
        self.current_frame = self.vision.capture()
        if self.current_frame is not None:
            
            # [★핵심 수정] VisionSystem의 ROI 정보를 Scanner에 동기화
            # UI에서 설정한 값이 Vision에만 있고 Scanner에는 없기 때문에 이를 전달해야 함
            if self.vision.minimap_roi:
                self.scanner.set_rois(self.vision.minimap_roi, self.vision.kill_roi)

            # 스캐너 업데이트
            self.player_pos = self.scanner.find_player(self.current_frame)
            self.scanner.update_skill_status(self.current_frame)

    def set_map_offset(self, x: int, y: int):
        self.map_processor.set_offset(x, y)

    