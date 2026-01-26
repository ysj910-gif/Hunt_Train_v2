# config.py

# [하드웨어 통신 설정]
# 윈도우: "COM3", "COM4" 등 (장치 관리자에서 확인)
# 맥/리눅스: "/dev/ttyUSB0" 등
# 이 값은 port_manager.py에 의해 자동으로 수정될 수 있습니다.
SERIAL_PORT = "COM8"  
BAUD_RATE = 115200

# [기타 설정]
# 필요한 경우 추가 설정 변수를 이곳에 정의합니다.