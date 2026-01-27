# engine/path_finder.py

import time
import math
import numpy as np
from utils.logger import logger, trace_logic
from utils.physics_utils import PhysicsUtils
from engine.advanced_pathfinder import AStarPathFinder

class PathFinder:
    def __init__(self, map_processor, physics_engine=None):
        self.map_processor = map_processor
        self.physics_engine = physics_engine
        
        self.astar = AStarPathFinder(map_processor, physics_engine)
        self.current_path_queue = [] 
        
        self.installed_objects = []
        self.install_skills = []
        
        self.current_target = None
        
        # [★핵심 1] 목표 고정(Sticky Target) 변수
        self.locked_install_target = None
        self.locked_skill_name = None
        
        # [★핵심 2] 재탐색 쿨타임 (너무 잦은 A* 방지)
        self.last_pathfind_time = 0
        self.PATHFIND_COOLDOWN = 0.5  # 0.5초 동안은 같은 타겟에 대해 재탐색 금지
        
        self.VISIT_THRESHOLD = 30.0

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
        
        # [설치 완료 시 고정 해제]
        if self.locked_skill_name == name:
            logger.info(f"🎯 목표 달성 완료: {name} 설치됨 -> 고정 해제")
            self.locked_install_target = None
            self.locked_skill_name = None
            
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

        # =================================================================
        # [★핵심 1] 목표 고정 로직 적용 (흔들림 방지)
        # =================================================================
        if self.locked_install_target and self.locked_skill_name:
            is_ready = install_ready_dict.get(self.locked_skill_name, False)
            is_installed = any(obj['name'] == self.locked_skill_name for obj in self.installed_objects)
            
            # 목표가 여전히 유효하다면(준비됨 + 미설치) -> 절대 바꾸지 않음
            if is_ready and not is_installed:
                return self.locked_install_target, "install_skill", self.locked_skill_name
            else:
                self.locked_install_target = None
                self.locked_skill_name = None

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
                # [새로운 최적 위치 발견 -> 고정]
                self.locked_install_target = best_spot
                self.locked_skill_name = next_skill['name']
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
        
        # 1. 목표 지점 선정 (여기서 고정된 타겟이 반환됨)
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
        # 큐가 비었거나, 타겟이 바뀌었을 때만 재탐색
        if not self.current_path_queue or self.current_target != target_pos:
            
            # [★핵심 2] 불필요한 빈번한 재탐색 방지 (같은 타겟이면 쿨타임 적용)
            now = time.time()
            if self.current_target == target_pos and (now - self.last_pathfind_time < self.PATHFIND_COOLDOWN):
                # 쿨타임 중인데 큐가 비었다면 -> 임시로 그냥 걷기/점프 공격
                return "move_and_attack", target_pos

            self.current_target = target_pos
            self.last_pathfind_time = now
            
            logger.debug(f"경로 재계산: {current_pos} -> {target_pos}")
            
            path = self.astar.find_path(current_pos, target_pos)
            if path:
                self.current_path_queue = path
                logger.debug(f"A* 경로 발견: {len(path)} steps")
            else:
                logger.warning("경로를 찾을 수 없음 (Fallback)")
                dy = abs(target_pos[1] - current_pos[1])
                dx = abs(target_pos[0] - current_pos[0])
                
                # Case 1: 같은 층 (dy < 8)
                if dy < 8: 
                    if mode == "install_skill":
                        return "move_to_install", target_pos
                    return "move_and_attack", target_pos

                # Case 2: 다른 층 (dy >= 8)
                else:
                    # 목표가 위에 있고 X좌표가 비슷하면 윗점프
                    if target_pos[1] < current_pos[1] and dx < 30:
                         return "execute_path", "up_jump"
                    return "move_to_install", target_pos

        # 3. 경로 실행
        if self.current_path_queue:
            next_action = self.current_path_queue.pop(0)
            return "execute_path", next_action 

        return "move_and_attack", target_pos