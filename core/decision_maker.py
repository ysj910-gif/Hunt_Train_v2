# core/decision_maker.py

import time
from enum import Enum, auto
from utils.logger import logger, trace_logic

# 신경망 컨트롤러
try:
    from core.neural_control import NeuralController
except ImportError:
    NeuralController = None

class BotState(Enum):
    IDLE = auto()
    MAPS = auto()
    COMBAT = auto()
    EMERGENCY = auto()

class DecisionMaker:
    """
    봇의 두뇌 클래스. 
    상태 판단, AI 추론, 그리고 Rule-based 전략을 결합하여 행동을 결정합니다.
    """
    def __init__(self, agent):
        self.agent = agent
        
        if NeuralController:
            self.neural_controller = NeuralController()
        else:
            self.neural_controller = None
            
        # 상태 추적 (Feature 계산용)
        self.prev_pos = None
        self.prev_time = time.time()

    def update(self):
        state = self.agent.state
        if state == BotState.IDLE:
            self._handle_idle()
        elif state == BotState.MAPS:
            self._handle_maps()
        elif state == BotState.COMBAT:
            self._handle_combat()
        elif state == BotState.EMERGENCY:
            self._handle_emergency()

    #@trace_logic
    def _handle_idle(self):
        self.agent.last_action = "Idle"
        self.agent.last_action_desc = "Waiting..."
        time.sleep(0.1)

    #@trace_logic
    def _handle_maps(self):
        if not self.agent.player_pos: return
        target_pos = self.agent.path_finder.find_next_patrol_target(self.agent.player_pos)
        
        if not target_pos:
            self.agent.state = BotState.COMBAT
            return

        target_x, target_y = target_pos
        self.agent.last_action = "Moving"
        self.agent.last_action_desc = f"To ({target_x}, {target_y})"

        success = self.agent.action_handler.move_x(
            target_x, 
            get_current_pos=lambda: self.agent.scanner.find_player(self.agent.vision.capture())
        )
        if not success:
            self.agent.state = BotState.EMERGENCY

    @trace_logic
    def _handle_combat(self):
        if not self.agent.player_pos: return

        # =================================================================
        # [전략 1] 설치기 스마트 설치 (Rule-based Override)
        # =================================================================
        install_name = "fountain"
        
        # 1. 쿨타임이 끝났는지 확인
        is_cooldown = self.agent.scanner.is_cooldown(install_name)
        
        # 2. 이미 맵 어딘가에 설치되어 있는지 확인 (PathFinder 메모리)
        # installed_objects = [{'name':..., 'pos':..., 'expiry':...}]
        installed_list = getattr(self.agent.path_finder, 'installed_objects', [])
        is_installed = any(obj['name'] == install_name for obj in installed_list)
        
        # 설치가 필요한 상황이라면? (쿨타임X AND 미설치)
        if not is_cooldown and not is_installed:
            # AI를 끄고 PathFinder의 '최적 위치 선정' 및 '이동 로직'을 따름
            # PathFinder는 내부적으로 가장 효율적인(스폰이 많은) 빈 곳을 찾아줌
            logger.info(f"✨ Strategy: Moving to optimal spot for {install_name}")
            self._execute_rule_combat() 
            return

        # =================================================================
        # [전략 2] AI 사냥 (Neural Control) + 구역 회피
        # =================================================================
        if self.neural_controller and self.neural_controller.loaded:
            self._execute_ai_combat()
            return

        # [전략 3] AI 없음 (Fallback)
        self._execute_rule_combat()

    def _execute_ai_combat(self):
        """딥러닝 모델 기반 전투 (설치기 구역 회피 로직 포함)"""
        current_pos = self.agent.player_pos
        now = time.time()
        
        # 1. Feature 계산 (속도)
        delta_x, delta_y = 0, 0
        if self.prev_pos:
            dt = now - self.prev_time
            if dt > 0:
                delta_x = (current_pos[0] - self.prev_pos[0])
                delta_y = (current_pos[1] - self.prev_pos[1])
        
        self.prev_pos = current_pos
        self.prev_time = now

        # 2. 발판 거리 계산
        px, py = current_pos
        curr_plat = self.agent.map_processor.find_current_platform(px, py)
        dist_left = 100
        dist_right = 100
        if curr_plat:
            dist_left = px - curr_plat['x_start']
            dist_right = curr_plat['x_end'] - px

        # 3. 상태 벡터 생성
        state_dict = {
            'player_x': px,
            'player_y': py,
            'delta_x': delta_x,
            'delta_y': delta_y,
            'dist_left': dist_left,
            'dist_right': dist_right,
            'inv_dist_up': 0,
            'inv_dist_down': 0,
            'inv_dist_left': 0,
            'inv_dist_right': 0,
            'kill_count': self.agent.scanner.current_kill_count,
            'ult_ready': 1 if not self.agent.scanner.is_cooldown('ultimate') else 0,
            'sub_ready': 0
        }
        
        # 4. AI 추론
        keys_to_press = self.neural_controller.predict(state_dict, threshold=0.4)
        
        if keys_to_press:
            # -----------------------------------------------------------
            # [구역 회피 로직] 설치기가 있는 곳으로는 가지 않도록 제어
            # -----------------------------------------------------------
            # 약 1초 뒤(또는 일정 거리) 이동할 위치가 커버 영역인지 체크
            check_dist = 50 # 50픽셀 앞을 미리 봄
            
            # 왼쪽으로 가려는데 그곳이 이미 커버된 구역이라면? -> 키 입력 취소
            if 'left' in keys_to_press:
                if self.agent.path_finder._is_point_covered(px - check_dist, py):
                    # logger.debug("🚫 Avoidance: Blocking LEFT (Covered Area)")
                    keys_to_press.remove('left')

            # 오른쪽으로 가려는데 그곳이 이미 커버된 구역이라면? -> 키 입력 취소
            if 'right' in keys_to_press:
                if self.agent.path_finder._is_point_covered(px + check_dist, py):
                    # logger.debug("🚫 Avoidance: Blocking RIGHT (Covered Area)")
                    keys_to_press.remove('right')
            
            # -----------------------------------------------------------
            # 키 입력 실행
            # -----------------------------------------------------------
            self.agent.last_action = "AI Control"
            self.agent.last_action_desc = str(keys_to_press)
            handler = self.agent.action_handler
            mapping = self.agent.key_mapping

            # [공격 키 수정] 'ctrl' 대신 설정된 스킬 키('r') 사용
            main_attack_key = mapping.get('main', 'r') 
            
            if 'left' in keys_to_press: handler.key_down('left')
            else: handler.key_up('left')
                
            if 'right' in keys_to_press: handler.key_down('right')
            else: handler.key_up('right')
                
            if 'jump' in keys_to_press: 
                jump_key = mapping.get('jump', 'alt')
                handler.press(jump_key)
                
            if 'attack' in keys_to_press: 
                handler.press(main_attack_key) # 수정된 공격 키
            
            if 'up' in keys_to_press: handler.key_down('up')
            else: handler.key_up('up')
            
            if 'down' in keys_to_press: handler.key_down('down')
            else: handler.key_up('down')
            
            if 'ultimate' in keys_to_press:
                ult_key = mapping.get('ultimate', '6')
                handler.press(ult_key)

    def _execute_rule_combat(self):
        """
        Rule-based 전투 로직 (설치기 설치용)
        PathFinder가 계산한 '최적의 위치'로 이동하여 설치하는 과정을 담당함
        """
        install_ready = not self.agent.scanner.is_cooldown("fountain") 
        
        # PathFinder에게 "어디로 가야 하니?" 물어봄
        # PathFinder는 설치가 필요하면 '설치 명당'으로 가는 경로를, 아니면 사냥 경로를 줌
        command, target = self.agent.path_finder.get_next_combat_step(self.agent.player_pos, install_ready)

        self.agent.last_action = command 
        self.agent.last_action_desc = str(target)
        handler = self.agent.action_handler
        mapping = self.agent.key_mapping 
        
        jump_key = mapping.get('jump', 'alt')
        attack_key = mapping.get('main', 'ctrl')

        # 설치 위치로 이동 중이거나, 설치 명령이 떨어졌을 때 실행됨
        if command == "execute_path":
            action = target
            
            if action == "up_jump":
                # [수정] 윗점프 입력 타이밍을 넉넉하게 보정
                handler.key_down("up")
                time.sleep(0.05) # 방향키 인식 대기
                handler.press(jump_key, duration=0.15) # 점프를 좀 더 길게 꾹 누름
                time.sleep(0.05)
                handler.key_up("up")
                time.sleep(0.8) # 체공 시간 대기 (0.7 -> 0.8로 약간 늘림)
            
            elif action == "down_jump":
                handler.key_down("down"); handler.press(jump_key); handler.key_up("down")
                time.sleep(0.5)
            elif action == "jump":
                handler.press(jump_key)
                time.sleep(0.5)
            elif action == "move_left":
                handler.press("left", duration=0.15)
            elif action == "move_right":
                handler.press("right", duration=0.15)

        elif command == "move_to_install":
            # 설치 위치로 걸어서 이동
            handler.move_x(target[0], lambda: self.agent.scanner.find_player(self.agent.vision.capture()))
            
        elif command == "install_skill":
            # 목적지 도착! 설치 실행
            skill_key = mapping.get("fountain", "4")
            handler.press(skill_key)
            # 설치 완료 사실을 PathFinder에 알려서 커버 영역으로 등록함
            self.agent.path_finder.update_install_status("fountain", *self.agent.player_pos) 
            logger.info("✅ Fountain Installed at Optimal Spot!")

        elif command == "move_and_attack":
            tx = target[0]; cx = self.agent.player_pos[0]
            direction = 'right' if tx > cx else 'left'
            handler.jump_shot(direction, jump_key=jump_key, attack_key=attack_key)
            
        elif command == "attack_on_spot":
            handler.jump_shot(None, jump_key=jump_key, attack_key=attack_key)

    def _handle_emergency(self):
        self.agent.last_action = "Recovering"
        self.agent.action_handler.emergency_stop()
        time.sleep(1.0)
        
        jump_key = self.agent.key_mapping.get('jump', 'alt')
        self.agent.action_handler.press(jump_key)
        time.sleep(0.5)
        
        self.agent.current_frame = self.agent.vision.capture()
        if self.agent.scanner.find_player(self.agent.current_frame):
            self.agent.state = BotState.IDLE 
        else:
            self.agent.stop()