# modules/map_creator.py
import json
import logging
from utils.logger import trace_logic, logger # [수정] 아키텍처 로거 사용

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
        self.new_portals = []   # 신규: 포탈 리스트
        self.new_ropes = []     # 신규: 밧줄 리스트
        self.new_map_portals = []  # [신규] 맵 이동 포탈 저장소
        self.new_spawns = [] 
        self.no_spawn_zones = []

        self.selected_type = None
        self.selected_index = None
        
        # [실행 취소 스택] (타입, 객체) 튜플을 저장하여 순서대로 취소
        self.action_history = []
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
    
    def add_no_spawn_zone(self, radius=50):
        pos = self.get_current_pos()
        if pos == (0, 0):
            return False, "현재 위치를 인식할 수 없습니다."
            
        zone = {
            "x": pos[0],
            "y": pos[1],
            "r": radius
        }
        self.no_spawn_zones.append(zone)
        self.action_history.append(("no_spawn", zone))
        
        logger.info(f"[MapCreator] No-Spawn Zone Added: {zone}")
        return True, f"현재 위치({pos})에 금지 구역이 설정되었습니다."

    def generate_spawns(self, total_monster_count):
        if not self.new_platforms:
            return False, "발판이 없습니다."

        # 1. 스폰 가능한 발판만 필터링 및 총 길이 계산
        valid_platforms = [p for p in self.new_platforms if p.get("allow_spawn", True)]
        if not valid_platforms:
            return False, "스폰 가능한 발판이 없습니다."

        total_length = sum((p["x_end"] - p["x_start"]) for p in valid_platforms)
        if total_length == 0:
            return False, "발판 길이 합이 0입니다."

        self.new_spawns = [] # 기존 스폰 초기화
        
        # 2. 각 발판별 비례 배분 및 배치
        current_count = 0
        
        for i, plat in enumerate(valid_platforms):
            p_len = plat["x_end"] - plat["x_start"]
            
            # 비례 배분 (마지막 발판에서 남은 수량 보정)
            if i == len(valid_platforms) - 1:
                count = total_monster_count - current_count
            else:
                ratio = p_len / total_length
                count = int(total_monster_count * ratio)
            
            current_count += count
            
            # 발판 내 균등 배치
            if count > 0:
                # 양 끝 10% 여유를 두고 배치
                margin = p_len * 0.1
                usable_len = p_len - (2 * margin)
                step = usable_len / (count + 1) if count > 0 else 0
                
                start_x = plat["x_start"] + margin
                
                for k in range(1, count + 1):
                    sx = int(start_x + (step * k))
                    sy = plat["y"] - 10 # 발판 약간 위
                    self.new_spawns.append({"x": sx, "y": sy})

        msg = f"총 {len(self.new_spawns)}개의 스폰 포인트가 {len(valid_platforms)}개 발판에 배치되었습니다."
        logger.info(msg)
        return True, msg

    def undo_last_platform(self):
        """마지막에 추가한 발판 취소"""
        if self.new_platforms:
            removed = self.new_platforms.pop()
            logger.info(f"[MapCreator] Undo Platform: {removed}")
            return True, removed
        return False, None
    
    def get_platform_count(self):
        return len(self.new_platforms)
    
    # MapCreator 클래스 내부에 추가
    def is_ready_to_add(self):
        """발판을 추가할 준비가 되었는지(시작점/종료점 설정 여부) 확인"""
        return self.temp_start_pos is not None and self.temp_end_pos is not None

    def _reset_temp_pos(self):
        """좌표 초기화 공통 함수"""
        self.temp_start_pos = None
        self.temp_end_pos = None
        self.clear_manual_pos()

    def add_platform(self, is_bottom=False):
        """임시 저장된 시작/종료점으로 발판을 생성합니다."""
        # 시작점/종료점 확인 (is_ready_to_add 메서드가 없으면 self.temp_start_pos 체크로 대체 가능)
        if hasattr(self, 'is_ready_to_add') and not self.is_ready_to_add():
             return False, "시작점과 종료점이 설정되지 않았습니다."
        elif not getattr(self, 'temp_start_pos', None) or not getattr(self, 'temp_end_pos', None):
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
            "y": y_avg,
            "type": "floor" if is_bottom else "platform" # [추가] 맨 아래 발판 여부 저장
        }
        
        self.new_platforms.append(new_plat)
        self.action_history.append(("platform", new_plat)) # 히스토리 기록
        self._reset_temp_pos()
        
        logger.info(f"[MapCreator] Platform Added (Bottom={is_bottom}): {new_plat}")
        return True, new_plat

    def add_portal(self):
        """[신규] 포탈 추가"""
        if not self.is_ready_to_add():
            return False, "시작점과 종료점이 설정되지 않았습니다."

        # 포탈은 시작점(src)에서 끝점(dst)으로 이동하는 연결 정보
        new_portal = {
            "src": self.temp_start_pos,
            "dst": self.temp_end_pos
        }

        self.new_portals.append(new_portal)
        self.action_history.append(("portal", new_portal))
        self._reset_temp_pos()

        logger.info(f"[MapCreator] Portal Added: {new_portal}")
        return True, new_portal

    def add_rope(self):
        """[신규] 밧줄 추가"""
        if not self.is_ready_to_add():
            return False, "시작점과 종료점이 설정되지 않았습니다."

        x1, y1 = self.temp_start_pos
        x2, y2 = self.temp_end_pos

        # 밧줄은 수직 구조물이므로 X는 평균값, Y는 위/아래 범위로 설정
        new_rope = {
            "x": int((x1 + x2) / 2),
            "y_top": min(y1, y2),
            "y_bottom": max(y1, y2)
        }

        self.new_ropes.append(new_rope)
        self.action_history.append(("rope", new_rope))
        self._reset_temp_pos()

        logger.info(f"[MapCreator] Rope Added: {new_rope}")
        return True, new_rope
    
    def add_map_portal(self, target_map_name):
        """[신규] 맵 이동 포탈 추가 (Start Point 위치 사용)"""
        if self.temp_start_pos is None:
            return False, "포탈 위치(시작점)가 설정되지 않았습니다."

        x, y = self.temp_start_pos
        
        # 맵 포탈 데이터 구조
        portal_data = {
            "x": x,
            "y": y,
            "to_map": target_map_name
        }

        self.new_map_portals.append(portal_data)
        self.action_history.append(("map_portal", portal_data))
        self._reset_temp_pos() # 위치 초기화

        logger.info(f"[MapCreator] Map Portal Added: {portal_data}")
        return True, portal_data

    def undo_last_action(self):
        """[수정] 실행 취소 로직에 map_portal 추가"""
        if not self.action_history:
            return False, "취소할 작업이 없습니다."

        action_type, item = self.action_history.pop()

        if action_type == "platform":
            if item in self.new_platforms: self.new_platforms.remove(item)
        elif action_type == "portal":
            if item in self.new_portals: self.new_portals.remove(item)
        elif action_type == "rope":
            if item in self.new_ropes: self.new_ropes.remove(item)
        elif action_type == "map_portal": # [신규]
            if item in self.new_map_portals: self.new_map_portals.remove(item)
            
        logger.info(f"[MapCreator] Undo {action_type}: {item}")
        return True, f"{action_type} 취소됨"

    def get_summary(self):
        """[수정] 요약 정보 갱신"""
        return (f"Plat: {len(self.new_platforms)} | Portal: {len(self.new_portals)} | "
                f"Rope: {len(self.new_ropes)} | MapP: {len(self.new_map_portals)}")
    
    def save_map_to_json(self, file_path):
        # [수정] 데이터 유무 체크
        if not (self.new_platforms or self.new_portals or self.new_ropes or self.new_map_portals):
            return False, "저장할 데이터가 없습니다."

        map_data = {
            "platforms": self.new_platforms,
            "portals": self.new_portals,
            "ropes": self.new_ropes,
            "map_portals": self.new_map_portals, # [신규] JSON 키 추가
            "spawns": []
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
        """모든 편집 데이터 초기화"""
        self.new_platforms = []
        self.new_portals = []
        self.new_ropes = []
        self.new_map_portals = []
        self.new_spawns = []
        self.no_spawn_zones = []
        self.action_history = []
        self._reset_temp_pos()
        
        # [신규] 디버깅 로그 추가
        logger.info("[MapCreator] All editor data cleared (Reset to blank state).")

    def load_from_json(self, file_path):
        """
        기존 맵 JSON 파일을 읽어서 편집 가능한 상태로 로드합니다.
        반환: (성공여부, 메시지)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            
            # 기존 데이터 초기화
            self.clear_data()
            
            # JSON에서 데이터 로드
            self.new_platforms = map_data.get("platforms", [])
            self.new_portals = map_data.get("portals", [])
            self.new_ropes = map_data.get("ropes", [])
            self.new_map_portals = map_data.get("map_portals", [])
            self.new_spawns = map_data.get("spawns", [])
            
            # 로드 통계
            summary = (f"발판: {len(self.new_platforms)}, "
                    f"포탈: {len(self.new_portals)}, "
                    f"밧줄: {len(self.new_ropes)}, "
                    f"맵포탈: {len(self.new_map_portals)}, "
                    f"스폰: {len(self.new_spawns)}")
            
            logger.info(f"[MapCreator] Loaded from {file_path} - {summary}")
            return True, summary
            
        except FileNotFoundError:
            msg = "파일을 찾을 수 없습니다."
            logger.error(f"[MapCreator] Load Error: {msg}")
            return False, msg
        except json.JSONDecodeError as e:
            msg = f"JSON 파싱 오류: {e}"
            logger.error(f"[MapCreator] Load Error: {msg}")
            return False, msg
        except Exception as e:
            msg = f"로드 실패: {e}"
            logger.error(f"[MapCreator] Load Error: {msg}")
            return False, msg
    
    # @trace_logic
    def select_object(self, obj_type, index):
        """UI에서 선택한 객체의 정보를 저장합니다."""
        self.selected_type = obj_type
        self.selected_index = index
        logger.debug(f"[MapCreator] Selected: {obj_type} #{index}")

    # [신규] 선택된 객체 삭제 메서드
    # @trace_logic
    def delete_selected(self):
        """현재 선택된 객체를 리스트에서 삭제합니다."""
        if self.selected_type is None or self.selected_index is None:
            return False, "선택된 객체가 없습니다."
        
        try:
            # 타입별 리스트에서 해당 인덱스 삭제
            if self.selected_type == "platform":
                del self.new_platforms[self.selected_index]
            elif self.selected_type == "portal":
                del self.new_portals[self.selected_index]
            elif self.selected_type == "rope":
                del self.new_ropes[self.selected_index]
            elif self.selected_type == "map_portal":
                del self.new_map_portals[self.selected_index]
            elif self.selected_type == "spawn":
                 del self.new_spawns[self.selected_index]
            else:
                return False, "삭제할 수 없는 객체 타입입니다."
            
            # 삭제 후 선택 초기화
            logger.info(f"[MapCreator] Deleted {self.selected_type} #{self.selected_index}")
            self.selected_type = None
            self.selected_index = None
            return True, "삭제되었습니다."
            
        except IndexError:
            return False, "이미 삭제되었거나 존재하지 않는 인덱스입니다."
        except Exception as e:
            return False, f"삭제 중 오류 발생: {str(e)}"
    