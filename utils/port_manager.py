# utils/port_manager.py

import serial.tools.list_ports
import re
import os

class PortManager:
    @staticmethod
    def find_arduino_port():
        """
        현재 연결된 COM 포트 중 아두이노(또는 USB Serial)로 추정되는 포트를 찾습니다.
        """
        # 연결된 모든 포트 리스트 가져오기
        ports = list(serial.tools.list_ports.comports())
        candidates = []
        
        print("🔍 연결된 포트 스캔 중...")
        for p in ports:
            # 디버깅용: 발견된 포트 정보 출력
            # print(f" - 발견됨: {p.device} ({p.description})")
            
            # 아두이노 또는 일반적인 시리얼 변환기 칩셋(CH340 등) 키워드 검색
            # 사용자의 환경에 따라 키워드를 추가/수정할 수 있습니다.
            keywords = ["Arduino", "CH340", "USB-SERIAL", "USB Serial", "Silicon Labs"]
            if any(k in p.description for k in keywords):
                candidates.append(p.device)
        
        if not candidates:
            print("⚠️ 아두이노로 추정되는 포트를 찾지 못했습니다.")
            # 후보가 없으면 연결된 포트가 1개일 경우 그것을 반환, 아니면 None
            if len(ports) == 1:
                print(f"👉 유일한 포트 {ports[0].device}를 선택합니다.")
                return ports[0].device
            return None
        
        # 가장 유력한 첫 번째 후보 반환
        return candidates[0]

    @staticmethod
    def update_config(config_file_path="config.py", variable_name="SERIAL_PORT"):
        """
        config.py 파일을 읽어서 포트 번호가 바뀌었으면 자동으로 수정합니다.
        """
        if not os.path.exists(config_file_path):
            print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file_path}")
            return None

        # 1. 새로운 포트 찾기
        new_port = PortManager.find_arduino_port()
        if not new_port:
            return None

        # 2. 파일 읽기
        with open(config_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 3. 기존 설정 찾기 (정규표현식 사용)
        # 예: SERIAL_PORT = "COM9" 패턴을 찾음
        pattern = rf'{variable_name}\s*=\s*["\'](COM\d+)["\']'
        match = re.search(pattern, content)

        if match:
            current_port = match.group(1)
            
            # 4. 포트가 다르면 파일 수정
            if current_port != new_port:
                print(f"🔄 포트 변경 감지! ({current_port} -> {new_port})")
                # 기존 내용을 새로운 포트로 치환
                new_content = re.sub(pattern, f'{variable_name} = "{new_port}"', content)
                
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"💾 {config_file_path} 파일이 업데이트되었습니다.")
            else:
                print(f"✅ 현재 포트 설정({current_port})이 유효합니다.")
        else:
            print(f"⚠️ {config_file_path}에서 {variable_name} 설정을 찾을 수 없습니다.")

        return new_port