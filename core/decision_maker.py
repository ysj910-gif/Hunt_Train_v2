# core/decision_maker.py

import time
import config 
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Optional

from utils.logger import logger, trace_logic

# 신경망 컨트롤러 (CombatState에서 사용)
try:
    from core.neural_control import NeuralController
except ImportError:
    NeuralController = None

class BotState(Enum):
    IDLE = auto()
    MAPS = auto()
    COMBAT = auto()
    EMERGENCY = auto()

# ==================================================================================
# 1. Abstract Base State
# ==================================================================================

class BaseState(ABC):
    @property
    @abstractmethod
    def state_enum(self) -> BotState:
        """해당 상태의 Enum 값을 반환"""
        pass

    @abstractmethod
    def execute(self, agent) -> "BaseState":
        """
        상태별 로직을 수행하고, 다음 상태(자신 혹은 새로운 상태)를 반환합니다.
        """
        pass

# ==================================================================================
# 2. Concrete States
# ==================================================================================

class IdleState(BaseState):
    state_enum = BotState.IDLE

    def execute(self, agent) -> BaseState:
        agent.last_action = "Idle"
        agent.last_action_desc = "Waiting..."
        time.sleep(0.1)
        # Idle 상태는 스스로 변하지 않고 외부(UI 등) 입력을 기다림
        return self


class MapsState(BaseState):
    state_enum = BotState.MAPS

    #@trace_logic
    def execute(self, agent) -> BaseState:
        if not agent.player_pos: 
            return self

        target_pos = agent.path_finder.find_next_patrol_target(agent.player_pos)
        
        # 순찰 목표가 없으면 전투 시작 (Maps -> Combat)
        if not target_pos:
            logger.info("🗺️ Map patrol finished. Switching to COMBAT.")
            return CombatState()

        target_x, target_y = target_pos
        agent.last_action = "Moving"
        agent.last_action_desc = f"To ({target_x}, {target_y})"

        success = agent.action_handler.move_x(
            target_x, 
            get_current_pos=lambda: agent.scanner.find_player(agent.vision.capture())
        )
        
        # 이동 실패 시 비상 상태 (Maps -> Emergency)
        if not success:
            logger.warning("⚠️ Movement failed. Switching to EMERGENCY.")
            return EmergencyState()

        return self


class CombatState(BaseState):
    state_enum = BotState.COMBAT

    def __init__(self):
        # Combat 상태 내부에서만 쓰이는 변수들 (캡슐화)
        self.prev_pos = None
        self.prev_time = time.time()

    #@trace_logic
    def execute(self, agent) -> BaseState:
        if not agent.player_pos: 
            return self

        # 1. 설치기 우선 설치 (Rule-based Override)
        if self._try_install_skill(agent):
            return self

        # 2. AI 전투 (Neural Control)
        # agent.brain을 통해 NeuralController에 접근
        if agent.brain.neural_controller and agent.brain.neural_controller.loaded:
            self._execute_ai_combat(agent)
            return self

        # 3. Rule-based 전투 (Fallback)
        self._execute_rule_combat(agent)
        return self

    def _try_install_skill(self, agent) -> bool:
        """설치기가 필요하면 설치 로직 수행 후 True 반환"""
        install_name = config.INSTALL_SKILL_NAME
        is_cooldown = agent.scanner.is_cooldown(install_name)
        installed_list = getattr(agent.path_finder, 'installed_objects', [])
        is_installed = any(obj['name'] == install_name for obj in installed_list)
        
        if not is_cooldown and not is_installed:
            logger.info(f"✨ Strategy: Moving to optimal spot for {install_name}")
            self._execute_rule_combat(agent) # 설치 위치 이동 로직은 Rule-based가 담당
            return True
        return False

    def _execute_ai_combat(self, agent):
        """AI 모델 기반 전투 로직"""
        current_pos = agent.player_pos
        now = time.time()
        
        # 속도 계산
        delta_x, delta_y = 0, 0
        if self.prev_pos:
            dt = now - self.prev_time
            if dt > 0:
                delta_x = (current_pos[0] - self.prev_pos[0])
                delta_y = (current_pos[1] - self.prev_pos[1])
        
        self.prev_pos = current_pos
        self.prev_time = now

        # 발판 정보
        px, py = current_pos
        curr_plat = agent.map_processor.find_current_platform(px, py)
        dist_left, dist_right = 100, 100
        if curr_plat:
            dist_left = px - curr_plat['x_start']
            dist_right = curr_plat['x_end'] - px

        # 상태 벡터 생성
        state_dict = {
            'player_x': px, 'player_y': py,
            'delta_x': delta_x, 'delta_y': delta_y,
            'dist_left': dist_left, 'dist_right': dist_right,
            'inv_dist_up': 0, 'inv_dist_down': 0,
            'inv_dist_left': 0, 'inv_dist_right': 0,
            'kill_count': agent.scanner.current_kill_count,
            'ult_ready': 1 if not agent.scanner.is_cooldown('ultimate') else 0,
            'sub_ready': 0
        }
        
        # 예측 및 키 입력
        keys_to_press = agent.brain.neural_controller.predict(
            state_dict, threshold=config.AI_CONFIDENCE_THRESHOLD
        )

        if keys_to_press:
            # 구역 회피 로직 (설치기 근처 접근 금지)
            check_dist = config.SAFETY_DISTANCE 
            if 'left' in keys_to_press and agent.path_finder._is_point_covered(px - check_dist, py):
                keys_to_press.remove('left')
            if 'right' in keys_to_press and agent.path_finder._is_point_covered(px + check_dist, py):
                keys_to_press.remove('right')
            
            # Action 수행
            agent.last_action = "AI Control"
            agent.last_action_desc = str(keys_to_press)
            self._apply_keys(agent, keys_to_press)

    def _apply_keys(self, agent, keys):
        handler = agent.action_handler
        mapping = agent.key_mapping
        main_attack_key = mapping.get('main', config.DEFAULT_KEYS['main'])

        if 'left' in keys: handler.key_down('left')
        else: handler.key_up('left')
        
        if 'right' in keys: handler.key_down('right')
        else: handler.key_up('right')
        
        if 'jump' in keys: handler.press(mapping.get('jump', config.DEFAULT_KEYS['jump']))
        if 'attack' in keys: handler.press(main_attack_key)
        
        if 'up' in keys: handler.key_down('up')
        else: handler.key_up('up')
        
        if 'down' in keys: handler.key_down('down')
        else: handler.key_up('down')
        
        if 'ultimate' in keys: handler.press(mapping.get('ultimate', config.DEFAULT_KEYS['ultimate']))

    def _execute_rule_combat(self, agent):
        """Rule-based (PathFinder 위임) 전투 로직"""
        install_sk = config.INSTALL_SKILL_NAME
        install_ready = not agent.scanner.is_cooldown(install_sk) 
        command, target = agent.path_finder.get_next_combat_step(agent.player_pos, install_ready)

        agent.last_action = command 
        agent.last_action_desc = str(target)
        
        handler = agent.action_handler
        mapping = agent.key_mapping
        # [수정] 기본 키값 config에서 가져오기
        jump_key = mapping.get('jump', config.DEFAULT_KEYS['jump'])
        attack_key = mapping.get('main', config.DEFAULT_KEYS['main'])
        install_key = mapping.get(install_sk, config.DEFAULT_KEYS['fountain'])

        if command == "execute_path":
            self._handle_path_action(handler, target, jump_key)
        elif command == "move_to_install":
            handler.move_x(target[0], lambda: agent.scanner.find_player(agent.vision.capture()))
        elif command == "install_skill":
            handler.press(install_key)
            agent.path_finder.update_install_status(install_sk, *agent.player_pos) 
            logger.info(f"✅ {install_sk} Installed!")
        elif command == "move_and_attack":
            direction = 'right' if target[0] > agent.player_pos[0] else 'left'
            handler.jump_shot(direction, jump_key=jump_key, attack_key=attack_key)
        elif command == "attack_on_spot":
            handler.jump_shot(None, jump_key=jump_key, attack_key=attack_key)

    def _handle_path_action(self, handler, action, jump_key):
        # [수정] 딜레이 시간들 config로 대체
        if action == "up_jump":
            handler.key_down("up"); time.sleep(0.05)
            handler.press(jump_key, duration=config.TIME_KEY_PRESS); time.sleep(0.05)
            handler.key_up("up"); time.sleep(config.TIME_UP_JUMP_WAIT)
        elif action == "down_jump":
            handler.key_down("down"); handler.press(jump_key); handler.key_up("down")
            time.sleep(config.TIME_DOWN_JUMP_WAIT)
        elif action == "jump":
            handler.press(jump_key); time.sleep(config.TIME_JUMP_DELAY)
        elif action == "move_left":
            handler.press("left", duration=config.TIME_KEY_PRESS)
        elif action == "move_right":
            handler.press("right", duration=config.TIME_KEY_PRESS)


class EmergencyState(BaseState):
    state_enum = BotState.EMERGENCY

    def execute(self, agent) -> BaseState:
        agent.last_action = "Recovering"
        agent.action_handler.emergency_stop()
        time.sleep(config.TIME_RECOVERY_WAIT)        
        
        # 복구 시도 (점프)
        jump_key = agent.key_mapping.get('jump', 'alt')
        agent.action_handler.press(jump_key)
        time.sleep(config.TIME_RECOVERY_WAIT)
        
        # 플레이어 확인 후 복구되면 IDLE로 전환
        agent.current_frame = agent.vision.capture()
        if agent.scanner.find_player(agent.current_frame):
            logger.info("✅ Recovered from Emergency. Returning to IDLE.")
            return IdleState()
        else:
            logger.critical("❌ Recovery failed. Stopping Agent.")
            agent.stop()
            return self

# ==================================================================================
# 3. Decision Maker (Context)
# ==================================================================================

class DecisionMaker:
    """
    봇의 두뇌 클래스.
    State Pattern을 사용하여 현재 상태(current_state)에 행동을 위임합니다.
    """
    def __init__(self, agent):
        self.agent = agent
        
        # 신경망 모델 로드 (CombatState에서 공유 사용)
        if NeuralController:
            self.neural_controller = NeuralController()
        else:
            self.neural_controller = None

        # 초기 상태 설정
        self.current_state: BaseState = IdleState()

    def update(self):
        """Main Loop에서 호출되는 메서드"""
        
        # 1. 외부 상태 변경 감지 (Sync: UI -> Logic)
        # 사용자가 GUI 버튼 등으로 agent.state를 강제로 변경했을 경우 대응
        if self.agent.state != self.current_state.state_enum:
            self._sync_state_from_enum(self.agent.state)

        # 2. 현재 상태 실행 및 다음 상태 반환
        next_state = self.current_state.execute(self.agent)

        # 3. 상태 전환 처리 (Sync: Logic -> UI)
        if next_state is not self.current_state:
            self._transition_to(next_state)

    def _sync_state_from_enum(self, state_enum: BotState):
        """Enum 값에 맞춰 상태 객체를 강제로 변경"""
        logger.info(f"🔄 Manual State Change detected: {state_enum}")
        if state_enum == BotState.IDLE:
            self.current_state = IdleState()
        elif state_enum == BotState.MAPS:
            self.current_state = MapsState()
        elif state_enum == BotState.COMBAT:
            self.current_state = CombatState()
        elif state_enum == BotState.EMERGENCY:
            self.current_state = EmergencyState()

    def _transition_to(self, new_state: BaseState):
        """내부 로직에 의해 상태가 변경될 때 호출"""
        logger.info(f"🔄 State Transition: {self.current_state.state_enum.name} -> {new_state.state_enum.name}")
        self.current_state = new_state
        self.agent.state = new_state.state_enum # Agent의 Enum 값도 업데이트 (UI 표시용)