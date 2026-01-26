# engine/advanced_pathfinder.py

import heapq
import math
import time
from utils.logger import logger

class PathNode:
    def __init__(self, x, y, platform_id, g=0, h=0, parent=None, action=None):
        self.x = x
        self.y = y
        self.platform_id = platform_id  # 현재 밟고 있는 발판 ID (-1: 공중)
        self.g = g  # 현재까지의 비용 (시간)
        self.h = h  # 목표까지의 추정 비용 (거리/속도)
        self.f = g + h
        self.parent = parent
        self.action = action  # 이 노드에 오기 위해 수행한 행동

    def __lt__(self, other):
        return self.f < other.f

class AStarPathFinder:
    def __init__(self, map_processor, physics_engine):
        self.map_processor = map_processor
        self.physics_engine = physics_engine
        self.MAX_SIM_FRAMES = 90  # 최대 1.5초 시뮬레이션
        self.GOAL_TOLERANCE = 30  # 목표 도달 인정 범위 (px)

    @trace_logic
    def find_path(self, start_pos, target_pos):
        """
        A* 알고리즘으로 경로 탐색
        Returns: [(action, duration), ...]
        """
        start_plat = self.map_processor.find_current_platform(*start_pos)
        s_id = self.map_processor.platforms.index(start_plat) if start_plat in self.map_processor.platforms else -1
        
        start_node = PathNode(start_pos[0], start_pos[1], s_id, 0, self._heuristic(start_pos, target_pos))
        
        open_list = []
        heapq.heappush(open_list, start_node)
        
        # 방문 체크: (platform_id, x_grid_10px)
        closed_set = set()

        while open_list:
            curr = heapq.heappop(open_list)

            # 목표 도달 확인 (거리 기준)
            if math.hypot(curr.x - target_pos[0], curr.y - target_pos[1]) < self.GOAL_TOLERANCE:
                return self._reconstruct_path(curr)

            state_key = (curr.platform_id, int(curr.x // 20))
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            # 가능한 행동 시뮬레이션
            # (행동명, 예상 소요 프레임)
            actions = ["move_left", "move_right", "jump", "up_jump", "down_jump"]
            if curr.platform_id != -1: # 땅에 있을 때만 점프 가능
                 pass
            else:
                 # 공중에서는 이동 제약이 있지만, 일단 단순화하여 땅에서 시작하는 것만 계산
                 continue

            for action in actions:
                # 가지치기: 목표 방향과 반대되는 이동은 후순위 (혹은 스킵)
                if "left" in action and curr.x < target_pos[0] - 50: continue
                if "right" in action and curr.x > target_pos[0] + 50: continue

                next_state = self._simulate_action(curr, action)
                if next_state:
                    nx, ny, nid, dur = next_state
                    new_g = curr.g + dur
                    new_h = self._heuristic((nx, ny), target_pos)
                    
                    neighbor = PathNode(nx, ny, nid, new_g, new_h, curr, action)
                    heapq.heappush(open_list, neighbor)
        
        return [] # 경로 없음

    def _heuristic(self, pos, target):
        return math.hypot(target[0] - pos[0], target[1] - pos[1]) / 10.0 # 거리 비용 가중치 조절

    def _simulate_action(self, node, action):
        """물리 엔진을 이용해 행동 결과 예측"""
        # 1. 초기 속도 설정 (PhysicsEngine이 있으면 사용, 없으면 하드코딩)
        vx, vy = 0, 0
        gravity = 5.0
        
        if self.physics_engine and self.physics_engine.is_loaded:
            # 행동 ID 매핑 필요 (임시 ID 사용)
            act_id = self._get_action_id(action)
            (p_vx, p_vy), p_g = self.physics_engine.predict_velocity(act_id, 1.0)
            vx, vy, gravity = p_vx, p_vy, p_g
        else:
            # Fallback 물리값
            if "jump" in action: vy = -65.0
            if "up_jump" in action: vy = -140.0
            if "move_left" in action: vx = -18.0
            if "move_right" in action: vx = 18.0
            if "down_jump" in action: vy = -10.0 # 살짝 뜸

        # 2. 궤적 시뮬레이션
        sim_x, sim_y = node.x, node.y
        
        for t in range(1, self.MAX_SIM_FRAMES):
            sim_x += vx
            sim_y += vy
            vy += gravity

            # 3. 착지 판정
            landed = self.map_processor.find_current_platform(sim_x, sim_y)
            if landed:
                # 하향 점프인 경우 현재 발판은 무시해야 함
                if "down_jump" in action and landed['y'] <= node.y + 10:
                    continue
                    
                plat_id = self.map_processor.platforms.index(landed)
                return (sim_x, landed['y'], plat_id, t)
            
            # 맵 이탈 체크 (Y축)
            if sim_y > 2000: break 

        # 단순히 걷기의 경우 발판 끝까지 갔는지 체크 (Move action)
        if "move" in action and not landed:
             # 시뮬레이션 끝났는데 바닥이 없으면(절벽) -> 떨어지는 경로도 노드로 추가 가능하나 일단 제외
             return None

        return None

    def _reconstruct_path(self, node):
        path = []
        curr = node
        while curr.parent:
            path.append(curr.action)
            curr = curr.parent
        return path[::-1]

    def _get_action_id(self, action):
        # 학습된 모델의 LabelEncoder 순서에 맞게 매핑
        mapping = {"move_left": 0, "move_right": 1, "jump": 2, "up_jump": 3, "down_jump": 4}
        return mapping.get(action, 0)