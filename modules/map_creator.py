# modules/map_creator.py
import json
import logging

# 로거 설정 (없으면 기본 로거 사용)
logger = logging.getLogger("MapCreator")
logger.setLevel(logging.INFO)

class MapCreator:
    """
    맵 제작 로직 클래스.
    - 실제 게임 좌표(Scanner)와 테스트용 수동 좌표(Manual Input)를 모두 지원합니다.
    """
    def __init__(self, agent=None):
        self.agent = agent
        self.new_platforms = []
        self.temp_start_pos = None
        self.temp_end_pos = None
        
        # 디버그용: 강제 좌표 주입 (유닛 테스트 또는 수동 입력용)
        self._manual_pos = None

    def set_manual_pos(self, x, y):
        """테스트/디버깅을 위해 현재 좌표를 강제로 설정합니다."""
        self._manual_pos = (int(x), int(y))
        logger.debug(f"[MapCreator] Manual Position Set: {self._manual_pos}")

    def clear_manual_pos(self):
        self._manual_pos = None

    def get_current_pos(self):
        """
        현재 좌표를 반환합니다.
        우선순위: 1.수동 입력값 -> 2.에이전트 스캐너 값 -> 3.(0,0)
        """
        # 1. 수동 입력 모드 확인
        if self._manual_pos:
            return self._manual_pos

        # 2. 에이전트 스캐너 확인
        if self.agent and hasattr(self.agent, 'scanner'):
            return self.agent.scanner.last_player_pos
            
        return (0, 0)

    def set_start_point(self):
        pos = self.get_current_pos()
        if pos == (0, 0):
            logger.warning("[MapCreator] Failed to set Start: Position is (0,0)")
            return False, pos
            
        self.temp_start_pos = pos
        logger.info(f"[MapCreator] Start Point Set: {pos}")
        return True, pos

    def set_end_point(self):
        pos = self.get_current_pos()
        if pos == (0, 0):
            logger.warning("[MapCreator] Failed to set End: Position is (0,0)")
            return False, pos
            
        self.temp_end_pos = pos
        logger.info(f"[MapCreator] End Point Set: {pos}")
        return True, pos

    def add_platform(self):
        """임시 저장된 시작/종료점으로 발판을 생성합니다."""
        if not self.temp_start_pos or not self.temp_end_pos:
            return False, "시작점과 종료점이 설정되지 않았습니다."

        x1, y1 = self.temp_start_pos
        x2, y2 = self.temp_end_pos

        # 좌표 정렬 및 계산
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_avg = int((y1 + y2) / 2) # 평지 가정
        
        # 유효성 검사 (폭이 너무 좁은 경우)
        if (x_end - x_start) < 5:
            return False, f"발판 길이가 너무 짧습니다 ({x_end - x_start}px)"

        new_plat = {
            "x_start": x_start,
            "x_end": x_end,
            "y": y_avg
        }
        
        self.new_platforms.append(new_plat)
        
        # 상태 초기화
        self.temp_start_pos = None
        self.temp_end_pos = None
        self.clear_manual_pos() # 발판 추가 후 수동 입력 해제 (선택사항)
        
        logger.info(f"[MapCreator] Platform Added: {new_plat}")
        return True, new_plat

    def undo_last_platform(self):
        """마지막에 추가한 발판 취소"""
        if self.new_platforms:
            removed = self.new_platforms.pop()
            logger.info(f"[MapCreator] Undo Platform: {removed}")
            return True, removed
        return False, None
    
    def get_platform_count(self):
        return len(self.new_platforms)

    def save_map_to_json(self, file_path):
        if not self.new_platforms:
            return False, "저장할 데이터가 없습니다."

        map_data = {
            "platforms": self.new_platforms,
            "spawns": [], 
            "portals": []
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=4, ensure_ascii=False)
            logger.info(f"[MapCreator] Saved to {file_path}")
            return True, "저장 성공"
        except Exception as e:
            logger.error(f"[MapCreator] Save Error: {e}")
            return False, str(e)
            
    def clear_data(self):
        self.new_platforms = []
        self.temp_start_pos = None
        self.temp_end_pos = None
        self._manual_pos = None