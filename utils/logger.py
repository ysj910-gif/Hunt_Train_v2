# utils/logger.py

import logging
import os
import json
import datetime
import functools
import time
import csv
from logging.handlers import RotatingFileHandler
import numpy as np  # <--- [필수] 이 줄이 꼭 있어야 합니다!

# 로그 저장 디렉토리 설정
LOG_DIR = r"C:\Temp\logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class CustomLogger:
    def __init__(self, name="MapleHunter"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # 상위 로거로 전파 방지
        self.tracing_enabled = False

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
        self.tracing_enabled = False
        
        # ✅ 트레이싱 대상 함수만 선택적으로 활성화
        self.traced_functions = set()  # 추가

    def enable_trace_for(self, *function_names):
        """특정 함수만 트레이싱 활성화"""
        self.traced_functions.update(function_names)
        self.tracing_enabled = True

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
        [수정됨] 파일 용량 제한 (5MB) 및 자동 로테이션 기능 추가
        """
        # 1. 파일 크기 체크 (5MB 제한)
        MAX_LOG_SIZE = 5 * 1024 * 1024
        
        if os.path.exists(self.logic_log_path):
            try:
                if os.path.getsize(self.logic_log_path) > MAX_LOG_SIZE:
                    backup_path = self.logic_log_path + ".backup"
                    # 기존 백업 삭제 후 현재 파일을 백업으로 이동
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(self.logic_log_path, backup_path)
            except Exception:
                pass # 파일 사용 중이면 무시

        # 2. 내용 구성
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "step": step,
            "state": state,
            "decision": decision,
            "reason": reason,
            "details": details
        }
        
        # 3. 파일 기록 (안전하게 문자열 변환)
        try:
            with open(self.logic_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"Log Error: {e}")

    def set_tracing(self, enabled: bool):
        self.tracing_enabled = enabled
        state = "ON" if enabled else "OFF"
        self.info(f"🔍 Logic Tracing Mode turned {state}")

    

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

# utils/logger.py 맨 아래의 trace_logic 함수를 이걸로 덮어쓰세요

def trace_logic(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # ✅ 1. 전역 스위치 체크
        if not logger.tracing_enabled:
            return func(*args, **kwargs)
        
        # ✅ 2. 함수명 필터링 (선택적 트레이싱)
        if logger.traced_functions and func.__name__ not in logger.traced_functions:
            return func(*args, **kwargs)
        
        # ✅ 3. 샘플링 (10번 중 1번만 기록)
        #if hash(time.time()) % 10 != 0:
        #    return func(*args, **kwargs)
        
        start_time = time.time()
        try:
            # 원본 함수 실행
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000

            # --- [핵심 수정] "무거운 객체(이미지)"를 문자열 변환에서 제외 ---
            
            # 1) 입력값(Args) 안전하게 변환
            safe_args = []
            for arg in args[1:]: # self 제외
                if isinstance(arg, np.ndarray): # Numpy 배열(이미지)인 경우
                    safe_args.append(f"<Img: {arg.shape}>")
                else:
                    s = str(arg)
                    safe_args.append(s[:50] + "..." if len(s) > 50 else s)
            
            params = ", ".join(safe_args)

            # 2) 결과값(Result) 안전하게 변환
            if isinstance(result, np.ndarray):
                result_str = f"<ImgResult: {result.shape}>"
            else:
                s = str(result)
                result_str = s[:100] + "..." if len(s) > 100 else s

            logger.debug(f"[TRACE] {func.__name__}({params}) -> {result_str} ({elapsed:.2f}ms)")
            return result

        except Exception as e:
            raise e
    return wrapper

class DataLogger:
    """
    [복구] 봇의 행동 데이터를 CSV로 저장하는 클래스 (v2 포팅)
    """
    def __init__(self, job_name, is_bot=False):
        # 데이터 저장 폴더 생성
        if not os.path.exists("data"):
            os.makedirs("data")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "Bot" if is_bot else "Human"
        
        self.filepath = f"data/{prefix}_{job_name}_{timestamp}.csv"
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        # 헤더 설정
        self.headers = [
            "timestamp", "entropy", "ult_ready", "sub_ready", 
            "action_name", "key_pressed", "player_x", "player_y", 
            "platform_id", "kill_count", "kill_reward",
            "dist_left", "dist_right", "job_class"
        ]
        self.writer.writerow(self.headers)
        self.last_kills = 0
        logger.info(f"✅ 데이터 수집 시작: {self.filepath}")

    def log_step(self, entropy, skill_manager, action_name, key_pressed, 
                 px, py, pid, current_kills, dist_left, dist_right, job_class):
        
        # 보상 계산
        reward = max(0, current_kills - self.last_kills)
        self.last_kills = current_kills
        
        # 스킬 상태 확인
        ult_ready = 1 if skill_manager.is_ready("ultimate") else 0
        sub_ready = 1 if skill_manager.is_ready("sub_attack") else 0
        
        # CSV 기록
        row = [
            time.time(),
            f"{entropy:.2f}",
            ult_ready,
            sub_ready,
            action_name,
            str(key_pressed), # 집합(Set)이나 문자열을 안전하게 변환
            px, py, pid,      
            current_kills,
            reward,
            round(dist_left, 1),
            round(dist_right, 1),
            job_class
        ]
        self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            logger.info("✅ 로그 파일 저장 완료")
