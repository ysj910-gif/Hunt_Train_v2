# utils/logger.py

import logging
import os
import json
import datetime
from logging.handlers import RotatingFileHandler

# 로그 저장 디렉토리 설정
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class CustomLogger:
    def __init__(self, name="MapleHunter"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # 상위 로거로 전파 방지

        # 포맷 설정
        # [시간] [레벨] [모듈명] 메시지
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%H:%M:%S'
        )

        # 1. 콘솔 핸들러 (실시간 출력용)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) # 콘솔엔 INFO 이상만 출력 (너무 시끄럽지 않게)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. 파일 핸들러 (전체 기록용 - Rotating)
        # 파일 하나당 5MB, 최대 5개 백업
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_handler = RotatingFileHandler(
            f"{LOG_DIR}/system_{today_str}.log", 
            maxBytes=5*1024*1024, 
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG) # 파일엔 모든 상세 로그 기록
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 3. 논리/판단 로그 전용 설정 (JSON 포맷)
        self.logic_log_path = f"{LOG_DIR}/decision_history_{today_str}.jsonl"

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    # [추가됨] 치명적 오류 기록용 메서드
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
        
    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def log_decision(self, step, state, decision, reason, **details):
        """
        봇의 의사결정 과정을 구조화된 JSON 데이터로 별도 저장합니다.
        
        Args:
            step (str): 현재 단계 (예: 'Vision', 'Navigator', 'Agent')
            state (str): 현재 봇의 상태 (예: 'SEARCH', 'COMBAT')
            decision (str): 내린 결정 (예: 'Move Left', 'Use Skill')
            reason (str): 결정의 근거 (예: 'Distance > 200', 'Cooldown Ready')
            **details: 기타 상세 수치 데이터
        """
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "step": step,
            "state": state,
            "decision": decision,
            "reason": reason,
            "details": details
        }
        
        # JSONL (Line-delimited JSON) 형식으로 저장 -> 나중에 분석하기 편함
        try:
            with open(self.logic_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write logic log: {e}")

# 전역에서 사용할 싱글톤 인스턴스 생성
# 다른 파일에서는: from utils.logger import logger
logger = CustomLogger()

if __name__ == "__main__":
    # 테스트 코드
    logger.info("시스템 초기화 시작...")
    logger.debug("디버그 메시지는 파일에만 기록됩니다.")
    logger.critical("치명적 오류 테스트 메시지입니다.") # 추가된 메서드 테스트
    
    # 의사결정 로그 테스트
    logger.log_decision(
        step="Navigator",
        state="PATROL",
        decision="Double Jump",
        reason="Target is far away",
        current_pos=(100, 200),
        target_pos=(500, 200),
        distance=400
    )
    
    logger.warning("주의: 몬스터 수가 너무 적습니다.")