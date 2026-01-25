#utils/physics_utils.py

import math
import numpy as np

class PhysicsUtils:
    """
    2D 게임 환경에서의 물리 연산 및 기하학적 계산을 돕는 유틸리티 클래스입니다.
    """

    @staticmethod
    def calc_distance(p1, p2):
        """
        두 점 사이의 유클리드 거리(직선 거리)를 반환합니다.
        
        Args:
            p1 (tuple): (x, y) 시작점
            p2 (tuple): (x, y) 끝점
        """
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def calc_manhattan_dist(p1, p2):
        """
        두 점 사이의 맨해튼 거리(|dx| + |dy|)를 반환합니다.
        점프 없이 이동해야 하는 격자형 맵에서의 거리 추정에 유용합니다.
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def calc_velocity(current_pos, prev_pos, dt=1.0):
        """
        이전 위치와 현재 위치의 차이를 통해 속도(변위)를 계산합니다.
        
        Args:
            current_pos (tuple): 현재 좌표 (x, y)
            prev_pos (tuple): 이전 프레임 좌표 (x, y)
            dt (float): 시간 변화량 (기본값 1.0 = 프레임 단위)
            
        Returns:
            tuple: (vx, vy)
        """
        if prev_pos is None:
            return 0.0, 0.0
        
        vx = (current_pos[0] - prev_pos[0]) / dt
        vy = (current_pos[1] - prev_pos[1]) / dt
        return vx, vy

    @staticmethod
    def is_stabilized(dx, dy, threshold_x=2.0, threshold_y=1.0):
        """
        오브젝트(플레이어)가 물리적으로 안정된 상태(착지/정지)인지 판별합니다.
        
        Args:
            dx, dy (float): 현재 속도
            threshold_x (float): X축 허용 오차 (미세한 미끄러짐 허용)
            threshold_y (float): Y축 허용 오차 (중력에 의한 낙하 감지용)
            
        Returns:
            bool: True면 착지/정지 상태
        """
        return abs(dx) <= threshold_x and abs(dy) <= threshold_y

    @staticmethod
    def is_point_in_range(target_pos, center_pos, ranges):
        """
        특정 점이 중심점 기준 사각형 범위(Up/Down/Left/Right) 내에 있는지 확인합니다.
        설치기(Fountain 등)의 커버 범위를 계산할 때 사용됩니다.
        
        Args:
            target_pos (tuple): 검사할 대상 좌표 (x, y)
            center_pos (tuple): 중심점 좌표 (cx, cy)
            ranges (dict): {'up': 100, 'down': 50, 'left': 200, 'right': 200} 형태의 범위
        """
        px, py = target_pos
        cx, cy = center_pos
        
        # 범위가 None이거나 비어있으면 기본값 0 처리
        up = ranges.get('up', 0)
        down = ranges.get('down', 0)
        left = ranges.get('left', 0)
        right = ranges.get('right', 0)
        
        x_min = cx - left
        x_max = cx + right
        y_min = cy - up
        y_max = cy + down
        
        return (x_min <= px <= x_max) and (y_min <= py <= y_max)

    @staticmethod
    def predict_landing_x(start_x, velocity_x, time_to_land):
        """
        단순 등속 운동을 가정하여 착지 예상 X 좌표를 계산합니다.
        """
        return start_x + (velocity_x * time_to_land)


class MovementTracker:
    """
    플레이어의 물리적 상태(위치, 속도, 고착 여부)를 추적하는 상태 관리자입니다.
    Agent 내부의 복잡한 변수들을 이 클래스로 위임할 수 있습니다.
    """
    def __init__(self):
        self.prev_pos = None
        self.last_move_time = 0
        self.stuck_count = 0
        self.dx = 0
        self.dy = 0
        
    def update(self, px, py, now_time):
        """매 프레임 호출하여 속도 및 상태 갱신"""
        if self.prev_pos is None:
            self.prev_pos = (px, py)
            self.last_move_time = now_time
            return
            
        # 속도 계산 (프레임 단위 변위)
        self.dx = px - self.prev_pos[0]
        self.dy = py - self.prev_pos[1]
        
        # 위치 갱신
        self.prev_pos = (px, py)
    
    def check_stuck(self, now_time, threshold_dist=20, timeout=2.0):
        """
        일정 시간 동안 이동 거리가 임계값 미만이면 Stuck으로 판정
        
        Args:
            threshold_dist: 이 거리보다 적게 움직이면 안 움직인 것으로 간주
            timeout: 이 시간(초) 동안 안 움직이면 Stuck
        """
        # 최근에 의미 있는 이동이 있었는지 확인
        dist = abs(self.dx) + abs(self.dy)
        if dist > 2: # 아주 미세한 움직임은 노이즈로 보고 무시
            self.last_move_time = now_time
            self.stuck_count = 0
            return False
            
        elapsed = now_time - self.last_move_time
        if elapsed > timeout:
            return True
            
        return False