# engine/path_finder.py

import time
import math
import numpy as np
from utils.logger import logger, trace_logic
from utils.physics_utils import PhysicsUtils
from engine.advanced_pathfinder import AStarPathFinder

class PathFinder:
    """
    [리팩토링 복원] 레거시 navigator.py의 전략적 타겟 선정 로직과
    신규 A* 물리 경로 탐색을 결합한 클래스입니다.
    """
    def __init__(self, map_processor, physics_engine=None):
        self.map_processor = map_processor
        self.physics_engine = physics_engine
        
        self.astar = AStarPathFinder(map_processor, physics_engine)
        self.current_path_queue = [] 
        
        self.installed_objects = []
        self.install_skills = []
        
        self.current_target = None
        self.last_strategy_time = 0
        
        self.VISIT_THRESHOLD = 30.0
        self.SCALE_RATIO = 1.0

    def register_install_skill(self, name, range_info, duration):
        self.install_skills.append({
            'name': name,
            'range': range_info,
            'duration': float(duration)
        })
        logger.info(f"설치기 등록: {name} (지속 {duration}s)")

    def update_install_status(self, name, x, y):
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
        for obj in self.installed_objects:
            ox, oy = obj['pos']
            rng = obj['range']
            left = ox - rng.get('left', 200)
            right = ox + rng.get('right', 200)
            top = oy - rng.get('up', 100)
            bottom = oy + rng.get('down', 50)
            
            if left <= px <= right and top <= py <= bottom:
                return True
        return False

    def _get_next_available_install(self, install_ready_dict):
        active_names = [obj['name'] for obj in self.installed_objects]
        for skill in self.install_skills:
            name = skill['name']
            is_ready = install_ready_dict.get(name, False)
            if is_ready and name not in active_names:
                return skill
        return None

    @trace_logic
    def get_optimum_target(self, player_pos, install_ready_dict):
        self._cleanup_expired_installs()
        spawns = self.map_processor.spawns
        if not spawns: return player_pos, "No Spawns", None

        px, py = player_pos
        
        # 1. 설치기 모드
        next_skill = self._get_next_available_install(install_ready_dict)
        
        if next_skill:
            best_score = -1
            best_spot = None
            for s in spawns:
                spos = (s['x'], s['y'])
                if self._is_point_covered(*spos): continue
                
                count = 0
                for other in spawns:
                    opos = (other['x'], other['y'])
                    if PhysicsUtils.calc_distance(spos, opos) < 300:
                        count += 1
                
                too_close = any(PhysicsUtils.calc_distance(spos, obj['pos']) < 200 for obj in self.installed_objects)
                if too_close: continue

                if count > best_score:
                    best_score = count
                    best_spot = spos
            
            if best_spot:
                return best_spot, "install_skill", next_skill['name']

        # 2. 순찰 모드
        best_score = float('inf')
        best_spot = None
        
        for s in spawns:
            spos = (s['x'], s['y'])
            dist = PhysicsUtils.calc_distance(player_pos, spos)
            if dist < self.VISIT_THRESHOLD: continue
            
            score = dist
            if self._is_point_covered(*spos):
                score += 5000.0
            
            if score < best_score:
                best_score = score
                best_spot = spos
        
        if best_spot:
            return best_spot, "move_and_attack", None
        
        return player_pos, "attack_on_spot", None

    @trace_logic
    def get_next_combat_step(self, current_pos, install_ready_dict):
        """BotAgent가 호출하는 메인 함수"""
        
        # 1. 목표 지점 선정
        target_pos, mode, skill_name = self.get_optimum_target(current_pos, install_ready_dict)
        
        if not target_pos:
            return "attack_on_spot", current_pos

        # 목표 도달 확인
        dist = PhysicsUtils.calc_distance(current_pos, target_pos)
        if dist < self.VISIT_THRESHOLD:
            if mode == "install_skill":
                return "install_skill", skill_name 
            else:
                self.current_path_queue = []
                return "attack_on_spot", current_pos

        # 2. 경로 탐색 (A*)
        if not self.current_path_queue or self.current_target != target_pos:
            self.current_target = target_pos
            logger.debug(f"경로 재계산: {current_pos} -> {target_pos}")
            
            path = self.astar.find_path(current_pos, target_pos)
            if path:
                self.current_path_queue = path
                logger.debug(f"A* 경로 발견: {len(path)} steps")
            else:
                logger.warning("경로를 찾을 수 없음 (Fallback)")
                return "move_and_attack", target_pos

        # 3. 경로 실행
        if self.current_path_queue:
            # [★핵심 수정] 실행할 행동을 큐에서 제거(pop)하여 다음 행동으로 넘어가게 함
            next_action = self.current_path_queue.pop(0)
            return "execute_path", next_action 

        return "move_and_attack", target_pos