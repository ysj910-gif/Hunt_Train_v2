# utils/human_input.py

import time
import random
import numpy as np
from utils.logger import logger

class HumanInput:
    """
    기계적인 입력을 방지하기 위해 통계적 분포(Ex-Gaussian, Beta)를 사용하여
    인간과 유사한 지연 시간(Delay)과 키 입력 지속 시간(Duration)을 생성하는 클래스입니다.
    """

    @staticmethod
    def ex_gaussian(mu: float, sigma: float, tau: float, min_val: float = 0.0, max_val: float = None) -> float:
        """
        Ex-Gaussian 분포를 사용하여 랜덤 값을 생성합니다.
        (인간의 반응 속도 모델링에 가장 적합한 분포)
        
        Args:
            mu (float): 정규 분포의 평균 (기본 반응 시간)
            sigma (float): 정규 분포의 표준편차 (반응 시간의 변동성)
            tau (float): 지수 분포의 스케일 (가끔 발생하는 느린 반응 - 꼬리 부분)
            min_val (float): 최소 반환값 (음수 방지)
            max_val (float): 최대 반환값 (너무 긴 딜레이 방지)
        """
        # 정규 분포 (Gaussian) 성분
        gaussian = np.random.normal(mu, sigma)
        # 지수 분포 (Exponential) 성분 - 롱테일(Long-tail) 효과
        exponential = np.random.exponential(tau)
        
        res = gaussian + exponential
        
        # 범위 제한 (Clipping)
        if min_val is not None:
            res = max(min_val, res)
        if max_val is not None:
            res = min(max_val, res)
            
        return res

    @staticmethod
    def beta_delay(target_time: float, variability: float = 0.2) -> float:
        """
        Beta 분포를 사용하여 목표 시간 근처에서 자연스러운 변동을 줍니다.
        Ex-Gaussian보다 꼬리가 짧고 안정적인 범위를 원할 때 사용합니다.
        
        Args:
            target_time (float): 목표 지연 시간 (초)
            variability (float): 변동 폭 (0.0 ~ 1.0)
        """
        # Beta 분포 파라미터 (alpha=2, beta=5 -> 오른쪽 꼬리가 약간 긴 형태)
        # 상황에 따라 파라미터 조정 가능
        val = np.random.beta(2, 5)
        
        # 0~1 사이의 값을 target_time 기준으로 스케일링
        # 예: val이 0.3이고 target이 0.1이면 -> 0.07 ~ 0.13 사이 분포 유도
        # 여기서는 단순화하여 target_time을 중심으로 variability만큼 퍼지게 함
        
        spread = target_time * variability
        base = target_time - (spread / 2)
        
        # Beta 분포 결과(0~1)를 spread 범위에 매핑
        randomized = base + (val * spread * 2) # *2는 skew 보정용 경험적 수치
        
        return max(0.01, randomized)

    @staticmethod
    def get_press_duration(base_duration: float = 0.07) -> float:
        """
        키를 누르고 있는 시간(Press Duration)을 생성합니다.
        짧은 탭(Tap)의 경우 Ex-Gaussian이 매우 자연스럽습니다.
        
        Args:
            base_duration: 목표 지속 시간 (기본 70ms)
        """
        # 설정: 평균적으로 base의 80% 시간, 나머지는 꼬리(Hesitation)로 처리
        mu = base_duration * 0.8
        sigma = base_duration * 0.1
        tau = base_duration * 0.2
        
        return HumanInput.ex_gaussian(mu, sigma, tau, min_val=0.02, max_val=0.5)

    @staticmethod
    def get_interval(base_interval: float = 0.1) -> float:
        """
        행동과 행동 사이의 대기 시간(Interval)을 생성합니다.
        """
        # 행동 사이에는 변동성이 더 큼
        mu = base_interval * 0.9
        sigma = base_interval * 0.2
        tau = base_interval * 0.5 # 꼬리를 길게 하여 가끔 멍때리는 효과
        
        return HumanInput.ex_gaussian(mu, sigma, tau, min_val=0.01, max_val=2.0)

    @staticmethod
    def human_sleep(base_time: float):
        """
        time.sleep() 대신 사용할 함수. base_time에 인간적인 노이즈를 추가하여 대기합니다.
        """
        if base_time <= 0:
            return
            
        # 0.1초 이하는 키 입력 지속시간으로 간주 (변동성 적게)
        if base_time < 0.1:
            sleep_time = HumanInput.get_press_duration(base_time)
        # 그 이상은 행동 간격으로 간주 (변동성 크게)
        else:
            sleep_time = HumanInput.get_interval(base_time)
            
        time.sleep(sleep_time)

# 테스트 코드
if __name__ == "__main__":
    print("--- Human Input Distribution Test ---")
    print(f"Target 0.1s (Press): {[round(HumanInput.get_press_duration(0.1), 3) for _ in range(5)]}")
    print(f"Target 0.5s (Wait) : {[round(HumanInput.get_interval(0.5), 3) for _ in range(5)]}")