# config.py

# [하드웨어 통신 설정]
SERIAL_PORT = "COM8"
BAUD_RATE = 115200

# [시스템 루프 및 성능 설정]
LOOP_INTERVAL = 0.05        # 메인 루프 최소 간격 (초)
FPS_SMOOTHING = 0.9         # FPS 계산 시 기존 값 가중치
WARNING_INTERVAL = 5        # 경고 로그 출력 간격 (초)
THREAD_TIMEOUT = 2.0        # 스레드 종료 대기 시간

# [비전 시스템 설정]
DEFAULT_RES_W = 1366        # 기본 해상도 너비
DEFAULT_RES_H = 768         # 기본 해상도 높이
WINDOW_RESTORE_DELAY = 0.5  # 최소화된 창 복구 대기 시간
WINDOW_ACTIVATE_DELAY = 0.2 # 창 활성화 대기 시간
SKILL_THRESH_RATIO = 0.7    # 스킬 쿨타임 자동 계산 비율 (현재 밝기 대비)
DEFAULT_SKILL_THRESH = 100.0 # 기본 스킬 임계값

# [AI 및 전투 로직 설정]
AI_CONFIDENCE_THRESHOLD = 0.4 # 신경망 예측 신뢰도 임계값
SAFETY_DISTANCE = 50          # 설치기/위험지역 회피 거리
INSTALL_SKILL_NAME = "fountain" # 설치기 스킬 이름 (키 매핑 조회용)

# [물리 엔진 설정]
PHYSICS_ACTION_DIM_DEFAULT = 16 # 물리 모델 기본 행동 차원 수

# [기본 키 매핑 (Key Mappings Fallback)]
# 사용자가 별도 설정을 하지 않았을 때 사용될 기본 키
DEFAULT_KEYS = {
    'main': 'ctrl',      # 기본 공격
    'jump': 'alt',       # 점프
    'ultimate': '6',     # 광역기/궁극기
    'fountain': '4',     # 설치기
    'rope': 'c'          # 로프 커넥트 등 (필요 시)
}

# [동작 타이밍 (Action Timings)]
TIME_KEY_PRESS = 0.15     # 일반적인 키 누름 시간
TIME_JUMP_DELAY = 0.5     # 점프 후 대기 시간
TIME_UP_JUMP_WAIT = 0.8   # 윗점프 후 체공 대기 시간
TIME_DOWN_JUMP_WAIT = 0.5 # 밑점프 후 대기 시간
TIME_RECOVERY_WAIT = 1.0  # 비상 복구 대기 시간