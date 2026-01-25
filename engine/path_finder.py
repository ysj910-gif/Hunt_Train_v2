# engine/path_finder.py

import time
from utils.logger import logger
from utils.physics_utils import PhysicsUtils

class PathFinder:
    """
    맵 데이터를 분석하여 캐릭터의 이동 경로와 전투 전략을 결정합니다.
    순찰 로직, 설치기 위치 선정, 추상화된 이동 커맨드 생성을 담당합니다.
    """
    def __init__(self, map_processor):
        self.map_processor = map_processor
        self.installed_objects = {}  # {name: {'pos': (x, y), 'expire_time': timestamp}}
        self.current_target_spawn = None
        self.patrol_index = 0

    def update_install_status(self, name, x, y, duration):
        """설치기(스킬)의 위치와 만료 시간을 갱신합니다."""
        self.installed_objects[name] = {
            'pos': (x, y),
            'expire_time': time.time() + duration
        }
        logger.log_decision(
            step="PathFinder",
            state="INSTALL",
            decision=f"Register {name}",
            reason="Skill deployed",
            x=x, y=y
        )

    def _is_area_covered(self, spawn_pos, range_px=150):
        """특정 스폰 지점이 현재 설치기에 의해 커버되고 있는지 확인합니다."""
        now = time.time()
        for name, info in list(self.installed_objects.items()):
            if now > info['expire_time']:
                del self.installed_objects[name]
                continue
            
            # 설치기 범위 내에 스폰 포인트가 있는지 체크
            if PhysicsUtils.calc_distance(spawn_pos, info['pos']) <= range_px:
                return True
        return False

    def find_next_patrol_target(self, current_pos):
        """
        설치기가 커버하지 못하는 구역 중 가장 효율적인 다음 목표물을 선정합니다.
       
        """
        spawns = self.map_processor.spawns
        if not spawns:
            return None

        # 1. 설치기에 의해 커버되지 않는 스폰 포인트 필터링
        uncovered_spawns = [s for s in spawns if not self._is_area_covered((s['x'], s['y']))]
        
        if not uncovered_spawns:
            # 모든 구역이 커버 중이라면 가장 먼 곳으로 이동하여 대기하거나 순찰
            uncovered_spawns = spawns

        # 2. 현재 위치에서 가장 가까운 미커버 스폰 지점 선택
        target = min(uncovered_spawns, key=lambda s: PhysicsUtils.calc_distance(current_pos, (s['x'], s['y'])))
        
        if self.current_target_spawn != target:
            self.current_target_spawn = target
            logger.log_decision(
                step="PathFinder",
                state="PATROL",
                decision="Change Target",
                reason="Moving to uncovered spawn area",
                target_pos=(target['x'], target['y'])
            )
            
        return (target['x'], target['y'])

    def get_move_command(self, current_pos):
        """
        현재 위치에서 목표 지점까지 가기 위한 추상화된 커맨드를 반환합니다.
       
        """
        tx, ty = self.find_next_patrol_target(current_pos)
        cx, cy = current_pos
        
        dx = tx - cx
        dy = ty - cy
        
        # 1. Y축 이동 판단 (복층 구조 대응)
        if abs(dy) > 10:  # 층 차이가 날 경우
            if dy > 0:
                return "down_jump", "Target is below current platform"
            else:
                return "up_jump", "Target is above current platform"
        
        # 2. X축 이동 판단
        if abs(dx) > 5:
            if dx > 0:
                return "move_right", f"Target X({tx}) is right of Current X({cx})"
            else:
                return "move_left", f"Target X({tx}) is left of Current X({cx})"
        
        return "stay", "Already at target area"

    def get_best_install_spot(self):
        """
        몬스터가 가장 밀집되어 있고, 현재 설치기가 없는 최적의 설치 장소를 계산합니다.
        """
        # 스폰 포인트 밀집도 분석 (단순 거리 기반 클러스터링 예시)
        best_spot = None
        max_density = -1
        
        for s in self.map_processor.spawns:
            pos = (s['x'], s['y'])
            if self._is_area_covered(pos):
                continue
                
            # 주변 100px 내의 다른 스폰 포인트 개수 계산
            density = sum(1 for other in self.map_processor.spawns 
                         if PhysicsUtils.calc_distance(pos, (other['x'], other['y'])) < 100)
            
            if density > max_density:
                max_density = density
                best_spot = pos
        
        if best_spot:
            logger.log_decision(
                step="PathFinder",
                state="STRATEGY",
                decision="Select Install Spot",
                reason=f"High density area (count: {max_density})",
                spot=best_spot
            )
        return best_spot