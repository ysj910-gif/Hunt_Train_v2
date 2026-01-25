#core\gen_manager.py

import time
from collections import deque
from utils.logger import logger

class GenManager:
    """
    몬스터 리젠(Gen) 주기를 관리하고 예측하는 클래스입니다.
    
    기능:
    1. 고정 주기(Default 7.5s) 기반 타이머
    2. 킬 카운트 변화를 감지하여 실제 젠 타이밍으로 자동 보정(Sync)
    3. 광역기 사용 등을 위한 타이밍 예측 신호 제공
    """
    
    def __init__(self):
        # 메이플스토리 일반적인 젠 주기 (서버 상태에 따라 7~8초 변동)
        self.DEFAULT_INTERVAL = 7.5
        self.current_interval = self.DEFAULT_INTERVAL
        
        # 타이밍 관리
        self.last_spawn_time = time.time()
        self.next_spawn_time = self.last_spawn_time + self.current_interval
        
        # 킬 카운트 추적용
        self.last_kill_count = -1
        
        # 동적 보정(Moving Average)을 위한 히스토리
        self.interval_history = deque(maxlen=10) 
        self.last_calibration_time = 0
        self.CALIBRATION_COOLDOWN = 5.0 # 너무 잦은 보정 방지 (한 젠에 한 번만)

    def update(self, current_kill_count: int):
        """
        매 프레임 호출되어 킬 카운트 변화를 감지하고 젠 주기를 보정합니다.
        
        Args:
            current_kill_count (int): 현재 누적 킬 카운트 (OCR 등에서 획득)
        """
        now = time.time()

        # 초기화 로직
        if self.last_kill_count == -1:
            self.last_kill_count = current_kill_count
            self.last_spawn_time = now
            self._update_next_spawn()
            return

        delta = current_kill_count - self.last_kill_count
        self.last_kill_count = current_kill_count

        # [핵심 로직] 유효한 전투(킬)가 발생했는지 확인
        # delta > 0: 몬스터가 죽음 -> 몬스터가 존재했음 -> 젠이 이미 되었음
        if delta > 0:
            time_since_last_calib = now - self.last_calibration_time
            
            # 쿨타임이 지났다면 이번 킬을 '젠 직후의 사냥'으로 간주하고 타이밍 동기화
            if time_since_last_calib > self.CALIBRATION_COOLDOWN:
                self._calibrate_cycle(now)

    def _calibrate_cycle(self, trigger_time):
        """실제 전투 발생 시각을 기준으로 사이클을 재설정 및 미세 조정"""
        # 이전 젠(추정)으로부터 실제 전투까지 걸린 시간 측정
        observed_interval = trigger_time - self.last_spawn_time
        
        # 터무니없는 값(예: 맵 이동 등으로 인한 30초 대기 등)은 통계에서 제외
        if 5.0 < observed_interval < 12.0:
            self.interval_history.append(observed_interval)
            
            # 이동 평균(Moving Average)으로 주기 미세 조정
            avg_interval = sum(self.interval_history) / len(self.interval_history)
            
            # 급격한 변화 방지를 위해 가중치 적용 (기존 7 : 새로운 데이터 3)
            self.current_interval = (self.current_interval * 0.7) + (avg_interval * 0.3)
            
            logger.debug(f"[GEN] Cycle calibrated: {observed_interval:.2f}s (Avg: {self.current_interval:.2f}s)")
        
        else:
            # 범위 밖이면 단순히 기준점만 리셋 (통계 반영 X)
            logger.debug(f"[GEN] Cycle reset (Out of range interval: {observed_interval:.2f}s)")

        # 기준점(앵커) 재설정
        self.last_spawn_time = trigger_time
        self.last_calibration_time = trigger_time
        self._update_next_spawn()

    def _update_next_spawn(self):
        self.next_spawn_time = self.last_spawn_time + self.current_interval

    def is_spawn_timing(self, buffer: float = 0.5) -> bool:
        """
        현재 시각이 예측된 젠 타임 근처인지 확인 (광역기 준비 신호)
        
        Args:
            buffer (float): 앞뒤 허용 오차 (초)
        """
        time_left = self.get_time_until_spawn()
        
        # 젠 시간이 거의 다 되었거나(양수), 방금 지났을 때(음수) 모두 포함
        # 예: buffer가 0.5면, -0.5 ~ +0.5 사이일 때 True
        if abs(time_left) <= buffer:
            return True
        
        # 혹은 젠이 2초 내로 임박했는지? (미리 자리 잡기용) -> 필요시 별도 메서드 분리
        return False

    def get_time_until_spawn(self) -> float:
        """
        다음 젠까지 남은 시간 반환
        Returns:
            float: 남은 시간 (초). 음수면 이미 시간이 지났음을 의미 (Overdue)
        """
        return self.next_spawn_time - time.time()

    def reset(self):
        """맵 이동 등으로 인한 초기화"""
        self.last_kill_count = -1
        self.interval_history.clear()
        self.current_interval = self.DEFAULT_INTERVAL
        logger.info("[GEN] Manager reset.")