# engine/map_processor.py

import json
import numpy as np
from utils.logger import logger
from utils.physics_utils import PhysicsUtils

class MapProcessor:
    """
    맵 데이터(JSON)를 파싱하고 발판, 스폰 포인트, 포탈 정보를 관리하는 클래스입니다.
    캐릭터의 위치와 맵 지형 간의 관계를 분석하여 Intelligence Layer에 제공합니다.
    """
    def __init__(self):
        self.platforms = []
        self.spawns = []
        self.portals = []
        self.map_name = ""
        
    def load_map(self, file_path: str) -> bool:
        """
        JSON 형식의 맵 파일을 로드하여 메모리에 저장합니다.
       
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.platforms = data.get("platforms", [])
            self.spawns = data.get("spawns", [])
            self.portals = data.get("portals", [])
            self.map_name = file_path.split('/')[-1]
            
            logger.info(f"맵 로드 완료: {self.map_name} (발판: {len(self.platforms)}, 스폰: {len(self.spawns)})")
            return True
        except Exception as e:
            logger.error(f"맵 로드 실패 ({file_path}): {e}")
            return False

    def find_current_platform(self, x: int, y: int):
        """
        현재 좌표(x, y)에서 캐릭터가 딛고 있는 가장 적절한 발판을 찾습니다.
        
        Args:
            x, y: 캐릭터의 현재 미니맵 좌표
        Returns:
            dict: 발판 정보 (없으면 None)
        """
        candidate_platforms = []
        
        for plat in self.platforms:
            # x축 범위 내에 있는지 확인
            if plat['x_start'] <= x <= plat['x_end']:
                # 캐릭터 발 밑에 있는 발판들 중 가장 가까운 것 (y값이 큰 것이 아래쪽)
                # 발판 y값은 보통 캐릭터 발 위치보다 약간 아래에 위치함
                if plat['y'] >= y - 5: # 오차 범위 5픽셀 허용
                    candidate_platforms.append(plat)
        
        if not candidate_platforms:
            return None
            
        # y값 차이가 가장 적은(가장 가까운 바닥) 발판 반환
        return min(candidate_platforms, key=lambda p: abs(p['y'] - y))

    def get_nearest_spawn(self, x: int, y: int):
        """
        현재 위치에서 가장 가까운 몬스터 스폰 포인트를 찾습니다.
       
        """
        if not self.spawns:
            return None
            
        return min(self.spawns, key=lambda s: PhysicsUtils.calc_distance((x, y), (s['x'], s['y'])))

    def get_ground_y(self, x: int, current_y: int) -> int:
        """
        특정 X 좌표에서 현재 높이 기준 가장 가까운 바닥의 Y 좌표를 반환합니다.
        좌표 보정 및 추락 판정에 사용됩니다.
        """
        plat = self.find_current_platform(x, current_y)
        if plat:
            return plat['y']
        
        # 발판을 못 찾은 경우 맵의 가장 낮은 발판(최하단 바닥) 반환
        if self.platforms:
            return max(self.platforms, key=lambda p: p['y'])['y']
            
        return current_y

    def is_on_edge(self, x: int, y: int, threshold: int = 5) -> str:
        """
        현재 발판의 끝에 도달했는지 확인합니다. (낙사 방지 및 이동 판단용)
        
        Returns:
            'left_edge', 'right_edge', 'middle'
        """
        plat = self.find_current_platform(x, y)
        if not plat:
            return 'none'
            
        if x <= plat['x_start'] + threshold:
            return 'left_edge'
        elif x >= plat['x_end'] - threshold:
            return 'right_edge'
        return 'middle'

    def get_all_platforms_at_y(self, y_target: int, tolerance: int = 2):
        """특정 높이에 있는 모든 발판 목록을 가져옵니다 (층 단위 분석용)"""
        return [p for p in self.platforms if abs(p['y'] - y_target) <= tolerance]