# collect_physics.py
import time
import os
import sys
import threading
import random  # 랜덤 모듈 추가1q   
from core.bot_agent import BotAgent
from utils.logger import logger

# 설정: 실험 반복 횟수
REPEAT_COUNT = 10 

def run_physics_experiments(agent):
    """물리 데이터 수집 시나리오 실행"""
    logger.info("🧪 물리 데이터 수집 실험을 시작합니다 (3초 후 시작)...")
    time.sleep(3)

    # 시나리오 1: 지상 마찰력 및 최고 속도 (Ground Friction & Max Speed)
    # 목표: 정지 -> 가속 -> 최고 속도 유지 -> 키 뗌 -> 감속 -> 정지
    for i in range(REPEAT_COUNT):
        logger.info(f"=== [Scenario 1] Ground Acceleration & Friction ({i+1}/{REPEAT_COUNT}) ===")
        
        # 1. 오른쪽으로 가속 (3초간 꾹 -> 최고 속도 도달)
        agent.recorder.set_scenario("Ground_Accel_Right_Max")
        agent.action_handler.handle_action("right", duration=3.0) 
        
        # 2. 키 떼고 자연 감속 (관성 측정)
        agent.recorder.set_scenario("Ground_Friction_Stop")
        time.sleep(1.5) # 완전히 멈출 때까지 대기

        # 3. 왼쪽으로 가속
        agent.recorder.set_scenario("Ground_Accel_Left_Max")
        agent.action_handler.handle_action("left", duration=3.0)
        
        # 4. 정지
        agent.recorder.set_scenario("Ground_Friction_Stop")
        time.sleep(1.5)

    # 시나리오 2: 점프 관성 (Jump Inertia)
    # 목표: 이동 중 점프 시 속도가 어떻게 유지되는지 측정
    for i in range(REPEAT_COUNT):
        logger.info(f"=== [Scenario 2] Jump Inertia ({i+1}/{REPEAT_COUNT}) ===")
        
        # 1. 오른쪽 이동 중 점프
        agent.recorder.set_scenario("Jump_Right_Inertia")
        # 키보드 누른 상태 시뮬레이션 (수동 구현 필요 시 BotAgent의 메서드 활용)
        # 여기서는 단순화를 위해 이동 -> 점프 순차 실행
        agent.action_handler.handle_action("right+jump", duration=0.8) 
        time.sleep(1.0) # 착지 대기

        # 2. 왼쪽 이동 중 점프
        agent.recorder.set_scenario("Jump_Left_Inertia")
        agent.action_handler.handle_action("left+jump", duration=0.8)
        time.sleep(1.0)

# 시나리오 3: 더블 점프 (Double Jump / Flash Jump)
    # 목표: 공중에서 추가 가속(Burst) 데이터 수집
    for i in range(REPEAT_COUNT):
        logger.info(f"=== [Scenario 3] Double Jump Physics ({i+1}/{REPEAT_COUNT}) ===")
        
        # 1. 오른쪽 더블 점프
        agent.recorder.set_scenario("Double_Jump_Right")
        
        # 동작 순서:
        # (1) 오른쪽 키를 누른 상태에서
        # (2) 점프(0.1초) -> (3) 약간 대기(공중 체류) -> (4) 다시 점프(발진)
        
        # 구현: 봇 에이전트는 복합 키 입력 처리가 가능해야 함
        # 여기서는 단순화를 위해 '오른쪽 이동'을 베이스로 깔고 점프 두 번 입력
        
        # Step 1: 점프해서 공중으로 띄움 (오른쪽 키 유지)
        agent.action_handler.handle_action("right+jump", duration=0.15)
        time.sleep(0.15) # 점프 후 아주 잠깐 대기 (타이밍 조절)
        
        # Step 2: 공중에서 재차 점프 (발진!)
        agent.action_handler.handle_action("right+jump", duration=0.15)
        
        # Step 3: 착지할 때까지 관찰 (오른쪽 키는 계속 누르거나 떼거나 실험)
        # 키를 떼고 관성으로 날아가는 것을 추천
        time.sleep(1.2) 

        # 2. 왼쪽 더블 점프 (대칭 데이터)
        agent.recorder.set_scenario("Double_Jump_Left")
        agent.action_handler.handle_action("left+jump", duration=0.15)
        time.sleep(0.15)
        agent.action_handler.handle_action("left+jump", duration=0.15)
        time.sleep(1.2)


    logger.info("✅ 모든 물리 실험 완료. 데이터를 저장합니다.")
    agent.stop()
    sys.exit()

def main():
    # BotAgent 초기화 (Vision, Scanner, Recorder 포함)
    agent = BotAgent()
    
    # 레코더가 물리 모드인지 확인 (Scenario 설정 기능이 있는지)
    if not hasattr(agent.recorder, "set_scenario"):
        logger.error("❌ DataRecorder가 물리 학습용 버전이 아닙니다.")
        return

    # 에이전트 스레드 시작 (화면 인식 및 레코딩 루프)
    agent_thread = threading.Thread(target=agent.run)
    agent_thread.start()

    # 실험 제어 스레드 시작
    experiment_thread = threading.Thread(target=run_physics_experiments, args=(agent,))
    experiment_thread.start()

def run_physics_experiments(agent):
    # ...
    for i in range(REPEAT_COUNT):
        # 1. 3.0초가 아니라 2.5 ~ 3.5초 사이 랜덤
        duration = random.uniform(2.5, 3.5)
        agent.action_handler.handle_action("right", duration=duration) 
        
        # 2. 대기 시간도 랜덤하게 (사람이 반응하는 것처럼)
        wait_time = random.uniform(1.2, 2.0)
        time.sleep(wait_time)

        # 3. 가끔은 짧게 끊어서 움직이기 (Jitter)
        if random.random() < 0.3: # 30% 확률로 딴짓
            agent.action_handler.handle_action("left", duration=0.1)
            time.sleep(0.5)

if __name__ == "__main__":
    main()