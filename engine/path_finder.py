# engine/path_finder.py

import time
import math
import numpy as np
from utils.logger import logger, trace_logic
from utils.physics_utils import PhysicsUtils
from engine.advanced_pathfinder import AStarPathFinder # [신규]

class PathFinder:
    """
    [리팩토링 복원] 레거시 navigator.py의 전략적 타겟 선정 로직과
    신규 A* 물리 경로 탐색을 결합한 클래스입니다.
    """
    def __init__(self, map_processor, physics_engine=None):
        self.map_processor = map_processor
        self.physics_engine = physics_engine
        
        # [신규] A* 경로 탐색기
        self.astar = AStarPathFinder(map_processor, physics_engine)
        self.current_path_queue = [] # [(action, target_pos), ...]
        
        # --- 복원된 상태 변수들 ---
        self.installed_objects = [] # [{'name':..., 'pos':..., 'expiry':...}]
        self.install_skills = []    # 사용할 설치기 설정 목록
        
        self.current_target = None
        self.last_strategy_time = 0
        
        # 설정값
        self.VISIT_THRESHOLD = 30.0
        self.SCALE_RATIO = 1.0 # 픽셀 단위 변환비 (필요시 조정)

    def register_install_skill(self, name, range_info, duration):
        """봇 초기화 시 사용할 설치기 등록"""
        self.install_skills.append({
            'name': name,
            'range': range_info, # {'up':.., 'down':..}
            'duration': float(duration)
        })
        logger.info(f"설치기 등록: {name} (지속 {duration}s)")

    def update_install_status(self, name, x, y):
        """설치기 사용 후 상태 업데이트"""
        # 해당 스킬 정보 찾기
        skill_info = next((s for s in self.install_skills if s['name'] == name), None)
        duration = skill_info['duration'] if skill_info else 60.0
        
        self.installed_objects.append({
            'name': name,
            'pos': (x, y),
            'expiry': time.time() + duration,
            'range': skill_info['range'] if skill_info else {}
        })
        logger.info(f"📍 설치기({name}) 활성화 @ ({x}, {y})")

    def _cleanup_expired_installs(self):
        now = time.time()
        self.installed_objects = [obj for obj in self.installed_objects if obj['expiry'] > now]

    def _is_point_covered(self, px, py):
        """어떤 좌표가 현재 활성화된 설치기 범위 내인지 확인"""
        for obj in self.installed_objects:
            ox, oy = obj['pos']
            rng = obj['range']
            # 범위 체크 (간단한 사각형)
            # 맵 좌표계 고려 (Y축: 위쪽이 작음)
            left = ox - rng.get('left', 200)
            right = ox + rng.get('right', 200)
            top = oy - rng.get('up', 100)    # 위쪽 (Y값 작음)
            bottom = oy + rng.get('down', 50) # 아래쪽 (Y값 큼)
            
            if left <= px <= right and top <= py <= bottom:
                return True
        return False

    def _get_next_available_install(self, install_ready_dict):
        """사용 가능한(쿨타임X, 현재 미설치) 설치기 찾기"""
        # 현재 맵에 깔려있는 스킬 이름들
        active_names = [obj['name'] for obj in self.installed_objects]
        
        for skill in self.install_skills:
            name = skill['name']
            # 1. 쿨타임이 준비되었는가? (Scanner 정보)
            is_ready = install_ready_dict.get(name, False)
            # 2. 이미 맵에 깔려있지 않은가? (중복 설치 방지)
            if is_ready and name not in active_names:
                return skill
        return None

    @trace_logic
    def get_optimum_target(self, player_pos, install_ready_dict):
        """
        [핵심 로직 복원] navigator.py의 점수 기반 타겟 선정
        """
        self._cleanup_expired_installs()
        spawns = self.map_processor.spawns
        if not spawns: return player_pos, "No Spawns"

        px, py = player_pos
        
        # 1. 설치기 모드 판별
        next_skill = self._get_next_available_install(install_ready_dict)
        
        target_candidates = []
        
        # 모드 A: 설치기 설치 (Install Mode)
        if next_skill:
            # 커버되지 않은 구역 중 가장 몬스터 밀집도가 높은(효율적인) 곳 찾기
            best_score = -1
            best_spot = None
            
            for s in spawns:
                spos = (s['x'], s['y'])
                if self._is_point_covered(*spos): continue
                
                # 주변 스폰 포인트 개수 카운트 (설치 효율)
                count = 0
                for other in spawns:
                    opos = (other['x'], other['y'])
                    if PhysicsUtils.calc_distance(spos, opos) < 300: # 반경 300px
                        count += 1
                
                # 기존 설치기와 너무 가까우면 제외
                too_close = any(PhysicsUtils.calc_distance(spos, obj['pos']) < 200 for obj in self.installed_objects)
                if too_close: continue

                if count > best_score:
                    best_score = count
                    best_spot = spos
            
            if best_spot:
                return best_spot, "install_skill", next_skill['name']

        # 모드 B: 일반 순찰 (Patrol Mode) - 회피 로직 포함
        best_score = float('inf') # 낮을수록 좋음 (Cost)
        best_spot = None
        
        for s in spawns:
            spos = (s['x'], s['y'])
            
            # 1. 기본 점수: 거리 (가까울수록 좋음)
            dist = PhysicsUtils.calc_distance(player_pos, spos)
            
            # 이미 도달한 곳은 제외 (히스테리시스)
            if dist < self.VISIT_THRESHOLD: continue
            
            score = dist
            
            # 2. [복원] 회피 로직 (Repulsion)
            # 설치기가 있는 곳은 봇이 직접 갈 필요가 없음 -> 점수 페널티 부여
            if self._is_point_covered(*spos):
                score += 5000.0 # 강력한 페널티
            
            if score < best_score:
                best_score = score
                best_spot = spos
        
        if best_spot:
            return best_spot, "move_and_attack", None
        
        return player_pos, "attack_on_spot", None

    @trace_logic
    def get_next_combat_step(self, current_pos, install_ready_dict):
        """
        BotAgent가 호출하는 메인 함수
        """
        # 1. 전략적 목표 지점 선정 (어디로 갈까?)
        target_pos, mode, skill_name = self.get_optimum_target(current_pos, install_ready_dict)
        
        if not target_pos:
            return "attack_on_spot", current_pos

        # 목표 도달 확인
        dist = PhysicsUtils.calc_distance(current_pos, target_pos)
        if dist < self.VISIT_THRESHOLD:
            if mode == "install_skill":
                return "install_skill", skill_name # 스킬명 반환
            else:
                # 도착했으면 큐 비우고 제자리 공격
                self.current_path_queue = []
                return "attack_on_spot", current_pos

        # 2. A* 경로 탐색 (어떻게 갈까?)
        # 경로가 없거나, 현재 쫓던 타겟이 변경되었으면 재계산
        if not self.current_path_queue or self.current_target != target_pos:
            self.current_target = target_pos
            logger.debug(f"경로 재계산: {current_pos} -> {target_pos}")
            
            path = self.astar.find_path(current_pos, target_pos)
            if path:
                self.current_path_queue = path
                logger.debug(f"A* 경로 발견: {len(path)} steps")
            else:
                logger.warning("경로를 찾을 수 없음 (Fallback)")
                # A* 실패 시 단순 이동 명령 반환 (Fallback)
                return "move_and_attack", target_pos

        # 3. 경로 실행
        if self.current_path_queue:
            next_action = self.current_path_queue[0]
            # 여기서는 단순히 다음 행동을 문자열로 반환
            # (ActionHandler가 이를 받아 처리)
            # 실제로는 ActionHandler가 해당 행동 완료 시점을 알려주거나 해야 함
            # 임시: 매 프레임 재계산을 방지하기 위해 큐를 유지하지만, 
            # 봇의 위치가 예상대로 바뀌었는지 체크하는 로직이 필요함.
            
            # 일단 단순화하여 첫 번째 스텝 반환
            return "execute_path", next_action # "jump", "move_right" 등

        return "move_and_attack", target_pos