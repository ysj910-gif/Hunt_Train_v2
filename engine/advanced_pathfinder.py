# engine/advanced_pathfinder.py

import heapq
import math
import time
from utils.logger import logger, trace_logic

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

            state_key = (curr.platform_id, int(curr.x // 2)) 
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            # 가능한 행동 시뮬레이션
            actions = ["move_left", "move_right", "jump", "up_jump", "down_jump"]
            
            if curr.platform_id != -1: # 땅에 있을 때만 점프 가능
                 pass
            else:
                 # 공중에서는 이동 제약이 있지만, 일단 단순화하여 땅에서 시작하는 것만 계산
                 continue

            for action in actions:
                # [수정] 가지치기 범위 축소 (미니맵 스케일에 맞춰 50 -> 10으로 변경)
                if "left" in action and curr.x < target_pos[0] - 10: continue
                if "right" in action and curr.x > target_pos[0] + 10: continue

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

    @trace_logic
    def _simulate_action(self, node, action):
        """물리 엔진을 이용해 행동 결과 예측 (미니맵 스케일 적용)"""
        
        # [핵심 수정] 스케일 조정: 미니맵이 실제 화면의 약 12% 크기라고 가정
        SCALE = 0.12  

        # 1. 초기 속도 설정 (미니맵 크기에 맞춰 스케일 다운)
        vx, vy = 0, 0
        gravity = 5.0 * SCALE # 중력 보정

        # AI 모델 대신 하드코딩된 물리값 사용 (안전성 확보 및 스케일 적용)
        # Fallback 물리값에 SCALE 적용
        if "jump" in action: vy = -65.0 * SCALE
        if "up_jump" in action: vy = -140.0 * SCALE
        if "move_left" in action: vx = -18.0 * SCALE
        if "move_right" in action: vx = 18.0 * SCALE
        if "down_jump" in action: vy = -10.0 * SCALE

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
                # [수정] 착지 허용 오차도 스케일에 맞춰 축소 (+10 -> +2)
                if "down_jump" in action and landed['y'] <= node.y + 2:
                    continue
                    
                plat_id = self.map_processor.platforms.index(landed)
                return (sim_x, landed['y'], plat_id, t)
            
            # [수정] 맵 이탈 체크 (Y축) - 미니맵 높이(약 84)를 고려하여 200 정도로 제한
            if sim_y > 200: break 

        # 단순히 걷기의 경우 발판 끝까지 갔는지 체크 (Move action)
        if "move" in action and not landed:
             # 절벽인 경우 떨어지는 로직을 추가할 수 있으나, 안전하게 None 반환
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