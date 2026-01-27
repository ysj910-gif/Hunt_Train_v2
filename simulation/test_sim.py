import sys
import os
import time

# -------------------------------------------------------------------------
# [설정] 부모 디렉터리(프로젝트 루트)를 모듈 경로에 추가
# 이렇게 하면 simulation 폴더 안에서도 engine, utils를 import 할 수 있습니다.
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from engine.map_processor import MapProcessor
    from engine.path_finder import PathFinder
    from utils.logger import logger
except ImportError as e:
    print(f"오류: 모듈을 불러올 수 없습니다. {e}")
    print(f"현재 경로: {current_dir}")
    print(f"부모 경로(추가됨): {parent_dir}")
    sys.exit(1)

def run_simulation():
    # 1. 맵 파일 경로 설정 (부모 폴더에 있는 파일 참조)
    map_file_name = "Corrected_Royal_Library_6.json"
    map_file_path = os.path.join(parent_dir, map_file_name)

    if not os.path.exists(map_file_path):
        print(f"오류: 맵 파일을 찾을 수 없습니다 -> {map_file_path}")
        return

    # 2. 맵 프로세서 & 패스파인더 초기화
    print(f"맵 로드 중: {map_file_name}")
    map_processor = MapProcessor()
    map_processor.load_map(map_file_path)
    
    # 물리 엔진 없이(A* 내부 계산 사용) 초기화
    path_finder = PathFinder(map_processor, physics_engine=None)
    
    # 3. 테스트 시나리오 설정
    # (로그에서 문제가 되었던 상황: 바닥에서 왼쪽 위로 이동)
    start_pos = (125, 66) 
    target_install_pos = (104, 29) 
    
    print(f"\n=== 🧪 시뮬레이션 시작 ===")
    print(f"📍 시작 위치: {start_pos}")
    print(f"🎯 목표 위치: {target_install_pos}")

    # 가상의 스킬 정보 등록
    install_ready_dict = {"fountain": True}
    path_finder.register_install_skill("fountain", {'left': 100, 'right': 100, 'up': 50, 'down': 50}, 60.0)

    # --- [Step 1] 첫 번째 요청 (목표 고정 확인) ---
    print(f"\n[Step 1] 경로 탐색 요청")
    command, action_or_pos = path_finder.get_next_combat_step(start_pos, install_ready_dict)
    
    print(f"👉 판단 결과: {command}")
    print(f"👉 세부 내용: {action_or_pos}")
    
    if path_finder.locked_install_target:
        print(f"✅ 목표 고정(Lock) 성공: {path_finder.locked_install_target}")
    else:
        print("❌ 목표 고정 실패")

    # --- [Step 2] 흔들림 방지 테스트 ---
    print(f"\n[Step 2] 위치 변경 후 재요청 (흔들림 테스트)")
    # 봇이 살짝 이동했다고 가정 (125 -> 123)
    current_pos = (123, 66)
    command, action_or_pos = path_finder.get_next_combat_step(current_pos, install_ready_dict)
    
    print(f"📍 현재 위치: {current_pos}")
    print(f"👉 판단 결과: {command}")
    
    # 목표가 유지되는지 확인
    if path_finder.current_target == target_install_pos:
        print(f"✅ 목표 유지 성공 (Sticky Target 동작 중)")
    else:
        print(f"❌ 목표가 변경됨: {path_finder.current_target}")

    # --- [Step 3] 쿨타임(스로틀링) 테스트 ---
    print(f"\n[Step 3] 0.1초 뒤 재요청 (재탐색 쿨타임 테스트)")
    time.sleep(0.1)
    command, action_or_pos = path_finder.get_next_combat_step(current_pos, install_ready_dict)
    print(f"👉 판단 결과: {command}")
    # 로그 레벨이 DEBUG라면 '경로 재계산' 로그가 안 찍혀야 정상

if __name__ == "__main__":
    run_simulation()