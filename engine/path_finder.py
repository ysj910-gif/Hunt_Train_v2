# engine/path_finder.py

import time
import math
from utils.logger import logger, trace_logic
from utils.physics_utils import PhysicsUtils

class PathFinder:
    """
    맵 데이터와 물리 엔진을 기반으로 최적의 사냥 경로를 계산하고 행동을 결정하는 클래스입니다.
    몬스터 인식 없이 맵의 스폰 포인트(Spawn Points)를 기반으로 '설치기 명당'과 '최단 순찰 경로'를 계산합니다.
    """
    def __init__(self, map_processor, physics_engine=None):
        self.map_processor = map_processor
        self.physics_engine = physics_engine
        
        # 설치기 상태 관리 {name: {'pos': (x, y), 'expire_time': ts}}
        self.installed_objects = {}  
        
        # 전략 관련 변수
        self.best_install_pos = None    # 계산된 최적의 설치기 위치
        self.patrol_route = []          # 계산된 최적 순찰 경로 (좌표 리스트)
        self.current_patrol_idx = 0     # 현재 순찰 중인 목표 인덱스
        
        # 물리 엔진 관련 설정
        # [주의] 학습된 모델의 LabelEncoder에서 'jump' 키가 할당된 ID를 확인하여 설정해야 합니다.
        self.JUMP_ACTION_ID = 2 

    def update_install_status(self, name, x, y, duration):
        """설치기(스킬) 사용 후 위치와 만료 시간을 등록합니다."""
        self.installed_objects[name] = {
            'pos': (x, y),
            'expire_time': time.time() + duration
        }
        logger.log_decision(
            step="PathFinder",
            state="INSTALL",
            decision=f"Registered {name}",
            reason=f"Expires in {duration}s",
            x=x, y=y
        )

    def _is_area_covered(self, spawn_pos, range_px=350):
        """
        특정 스폰 지점이 현재 설치기에 의해 커버되고 있는지 확인합니다.
        (설치기의 공격 범위를 고려하여 range_px 조정 필요)
        """
        now = time.time()
        # 만료된 설치기 제거
        for name, info in list(self.installed_objects.items()):
            if now > info['expire_time']:
                del self.installed_objects[name]
                continue
            
            # 거리 체크
            if PhysicsUtils.calc_distance(spawn_pos, info['pos']) <= range_px:
                return True
        return False

    @trace_logic
    def calculate_optimal_strategy(self, install_range_px=300):
        """
        [핵심 로직] 전체 맵을 분석하여 '최적의 설치 장소'와 '나머지 최단 순찰 경로'를 계산합니다.
        알고리즘:
        1. 모든 스폰 포인트를 설치기 후보지로 가정.
        2. 각 후보지에 설치했을 때, 커버되지 않는 나머지 포인트들을 방문하는 총 이동 거리(Total Distance) 계산.
        3. 이동 거리가 가장 짧은 시나리오를 채택.
        """
        spawns = self.map_processor.spawns
        if not spawns:
            return

        logger.info(f"전략 수립 시작 (스폰 포인트: {len(spawns)}개)...")

        best_score = float('inf') # 낮을수록 좋음 (총 이동 거리)
        best_setup = None         # (install_pos, patrol_route_list)

        # 1. 모든 스폰 포인트를 설치 후보로 시뮬레이션
        for i, candidate in enumerate(spawns):
            # install_pos: 설치기 후보 위치
            install_pos = (candidate['x'], candidate['y'])
            
            # 커버되지 않는 나머지 포인트들 추출
            uncovered_points = []
            for s in spawns:
                s_pos = (s['x'], s['y'])
                # 설치기 범위 밖이면 순찰 대상
                if PhysicsUtils.calc_distance(install_pos, s_pos) > install_range_px:
                    uncovered_points.append(s_pos)

            # 나머지 포인트들이 없으면(설치기 하나로 전맵 커버) 이 위치가 최적
            if not uncovered_points:
                best_score = 0
                best_setup = (install_pos, [])
                break

            # 2. Nearest Neighbor 알고리즘으로 순찰 경로 생성 (Greedy)
            # 시작점은 설치기 위치라고 가정 (설치 후 출발)
            current_node = install_pos
            total_dist = 0
            route = []
            to_visit = uncovered_points[:] # 리스트 복사

            while to_visit:
                # 현재 위치에서 가장 가까운 다음 지점 찾기
                next_node = min(to_visit, key=lambda p: PhysicsUtils.calc_distance(current_node, p))
                
                dist = PhysicsUtils.calc_distance(current_node, next_node)
                total_dist += dist
                
                route.append(next_node)
                to_visit.remove(next_node)
                current_node = next_node

            # 3. 점수 비교 (총 이동 거리가 가장 짧은 것 선택)
            if total_dist < best_score:
                best_score = total_dist
                best_setup = (install_pos, route)

        # 4. 최적값 확정 및 저장
        if best_setup:
            self.best_install_pos = best_setup[0]
            self.patrol_route = best_setup[1]
            self.current_patrol_idx = 0
            
            logger.info(f"✅ 전략 수립 완료!")
            logger.info(f" - 설치 위치: {self.best_install_pos}")
            logger.info(f" - 순찰 경로: {len(self.patrol_route)}개 지점 (총 거리: {best_score:.1f}px)")
        else:
            logger.warning("전략 수립 실패: 스폰 포인트가 없습니다.")

    @trace_logic
    def get_next_combat_step(self, current_pos, is_install_ready):
        """
        BotAgent가 매 프레임 호출하여 '다음 행동(Command)'을 받아가는 함수입니다.
        
        Returns:
            command (str): 행동 명령 ('move_to_install', 'install_skill', 'move_and_attack', 'attack_on_spot')
            target (tuple): 목표 좌표 (x, y) 또는 None
        """
        # 1. 전략이 수립되지 않았다면 계산
        if not self.best_install_pos:
            self.calculate_optimal_strategy()
            return "calculating", None

        # 2. 설치기 로직
        # 설치 쿨타임이 찼고(Ready), 현재 맵에 설치된 게 없다면 -> 설치하러 감
        if is_install_ready and not self.installed_objects:
            dist = PhysicsUtils.calc_distance(current_pos, self.best_install_pos)
            
            # 설치 위치 도착 (오차 30px 이내)
            if dist < 30:
                return "install_skill", self.best_install_pos
            
            # 설치 위치로 이동
            return "move_to_install", self.best_install_pos

        # 3. 순찰 로직 (설치기가 이미 있거나 쿨타임 중일 때)
        if not self.patrol_route:
            # 순찰할 곳이 없으면(설치기가 전맵 커버 등) 설치기 근처나 제자리에서 대기/공격
            return "attack_on_spot", current_pos

        target = self.patrol_route[self.current_patrol_idx]
        dist = PhysicsUtils.calc_distance(current_pos, target)

        # 목표 지점 도착 (오차 30px 이내)
        if dist < 30:
            # 다음 목표로 인덱스 변경 (순환 구조)
            self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_route)
            next_target = self.patrol_route[self.current_patrol_idx]
            
            # 도착했으므로 즉시 다음 지점을 향해 공격하며 이동
            return "move_and_attack", next_target
        
        # 목표를 향해 이동하면서 공격
        return "move_and_attack", target

    def _can_reach_with_jump(self, start_pos, target_pos):
        """
        물리 엔진을 사용하여 일반 점프로 목표 지점에 도달 가능한지 예측합니다.
        (PhysicsEngine이 올바르게 로드되어 있어야 함)
        """
        if not self.physics_engine or not self.physics_engine.is_loaded:
            return False

        cx, cy = start_pos
        tx, ty = target_pos
        
        # 1. 물리 엔진 예측 요청
        # (주의: 학습된 모델의 출력과 PhysicsEngine.predict_velocity의 반환값이 일치해야 함)
        pred = self.physics_engine.predict_velocity(action_idx=self.JUMP_ACTION_ID, is_ground=1.0)
        if not pred:
            return False
            
        vx, vy = pred[0]  # 초기 속도 (vx, vy)
        gravity = pred[1] # 중력
        
        # 2. 궤적 시뮬레이션 (약 1초/60프레임)
        sim_x, sim_y = cx, cy
        
        for _ in range(60): 
            sim_x += vx
            sim_y += vy
            vy += gravity # 중력 가속도 적용
            
            # 판정: 목표 높이보다 높게 올라갔고, 수평 거리도 닿는가?
            # (화면 좌표계: y가 작을수록 위쪽)
            if sim_y <= ty and abs(sim_x - tx) < 30: 
                return True
                    
            # 바닥보다 아래로 떨어지면 실패
            if sim_y > cy + 50: 
                break
                
        return False

    def get_move_command(self, current_pos, target_pos):
        """
        ActionHandler를 위한 저수준 이동 명령 생성
        (get_next_combat_step에서 반환된 target_pos를 받아 처리)
        """
        if not target_pos:
            return "stay", "No Target"

        cx, cy = current_pos
        tx, ty = target_pos
        
        dx = tx - cx
        dy = ty - cy
        
        # 1. Y축 이동 판단 (높이 차이가 클 때)
        if abs(dy) > 30:
            if dy > 0: # 목표가 아래에 있음
                return "down_jump", "Target is below"
            else: # 목표가 위에 있음
                # 물리 엔진으로 닿을 수 있는지 확인
                if self._can_reach_with_jump(current_pos, target_pos):
                    # 닿을 수 있다면 방향만 잡고 점프
                    direction = "right_jump" if dx > 0 else "left_jump"
                    return direction, "Physics reachable"
                
                return "up_jump", "Target is too high (Rope needed)"
        
        # 2. X축 이동
        if abs(dx) > 10:
            if dx > 0:
                return "move_right", f"Target is right ({int(dx)})"
            else:
                return "move_left", f"Target is left ({int(dx)})"
        
        return "stay", "At target"

    # 호환성 유지를 위한 메서드 (기존 코드에서 호출할 경우 대비)
    def find_next_patrol_target(self, current_pos):
        if not self.patrol_route:
            return None
        return self.patrol_route[self.current_patrol_idx]