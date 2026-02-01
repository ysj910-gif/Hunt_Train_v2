# engine/map_processor.py

import json
import numpy as np
from utils.logger import logger, trace_logic
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
        
        # UI에서 설정한 오프셋 값을 저장할 변수
        self.offset_x = 0
        self.offset_y = 0
        
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

    def set_offset(self, x: int, y: int):
        """
        UI에서 조정된 맵 오프셋 값을 업데이트합니다.
        """
        self.offset_x = x
        self.offset_y = y

    def find_current_platform(self, x: int, y: int):
        """
        현재 좌표(x, y)에서 캐릭터가 '실제로 밟고 있는' 발판을 찾습니다.
        """
        chk_x = x - self.offset_x
        chk_y = y - self.offset_y

        candidate_platforms = []
        
        for idx, plat in enumerate(self.platforms):
            # 1. X축 범위 검사
            if plat['x_start'] <= chk_x <= plat['x_end']:
                
                # 2. Y축 정밀 검사 ("위로만 3픽셀")
                # diff = 발판높이 - 캐릭터발높이
                # diff > 0: 캐릭터가 발판 위에 떠 있음 (Jump/Fall)
                # diff < 0: 캐릭터가 발판 아래로 잠김 (오차)
                diff = plat['y'] - chk_y
                
                # [수정된 조건]
                # -3 <= diff: 발이 발판 아래로 3픽셀까지 잠기는 것 허용 (오차 보정)
                # diff <= 3:  발이 발판 위로 3픽셀까지만 떠 있는 것 허용 (그 이상은 점프 중)
                if -3 <= diff <= 3:
                    candidate_platforms.append((abs(diff), idx, plat))
        
        step_name = "MapProcessor"
        state_name = "LOCATE_PLATFORM"

        # 후보가 없으면 "공중(Air)" 상태
        if not candidate_platforms:
            #logger.log_decision(
            #    step=step_name,
            #    state=state_name,
            #    decision="FAIL", 
            #    reason="In Air (Diff > 3px)",
            #    current_pos_adj=(chk_x, chk_y),
            #    check_y_threshold="-3 <= diff <= 3"
            #)
            return None
            
        # 가장 가까운 발판 선택
        best_diff, best_idx, best_plat = min(candidate_platforms, key=lambda p: p[0])

        #logger.log_decision(
        #    step=step_name,
        #    state=state_name,
        #    decision=f"Platform_Idx_{best_idx}",
        #    reason=f"Landed (Diff: {best_diff})",
        #    current_pos_adj=(chk_x, chk_y),
        #    selected_plat_y=best_plat['y']
        #)

        return best_plat

    def get_nearest_spawn(self, x: int, y: int):
        if not self.spawns:
            return None
        
        # 여기도 부호 변경 (- offset)
        chk_x = x - self.offset_x
        chk_y = y - self.offset_y
            
        return min(self.spawns, key=lambda s: PhysicsUtils.calc_distance((chk_x, chk_y), (s['x'], s['y'])))

    def get_ground_y(self, x: int, current_y: int) -> int:
        plat = self.find_current_platform(x, current_y)
        if plat:
            return plat['y']
        
        if self.platforms:
            return max(self.platforms, key=lambda p: p['y'])['y']
            
        # 못 찾으면 보정된 좌표 반환 (부호 변경 확인)
        return current_y - self.offset_y

    def is_on_edge(self, x: int, y: int, threshold: int = 5) -> str:
        plat = self.find_current_platform(x, y)
        if not plat:
            return 'none'
        
        # 부호 변경 (- offset)
        chk_x = x - self.offset_x
            
        if chk_x <= plat['x_start'] + threshold:
            return 'left_edge'
        elif chk_x >= plat['x_end'] - threshold:
            return 'right_edge'
        return 'middle'

    def get_all_platforms_at_y(self, y_target: int, tolerance: int = 2):
        return [p for p in self.platforms if abs(p['y'] - y_target) <= tolerance]
    
    def unload_map(self):
        """로드된 맵 데이터를 초기화합니다."""
        self.platforms = []
        self.spawns = []
        self.portals = []
        self.map_name = ""
        self.offset_x = 0
        self.offset_y = 0
        logger.info("MapProcessor: Map data unloaded (Reset to empty).")