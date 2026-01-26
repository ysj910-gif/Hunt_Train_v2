import sys
import os
import tkinter as tk
import ctypes
import traceback

# 프로젝트 루트 경로를 sys.path에 추가 (모듈 import 오류 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger
from core.bot_agent import BotAgent
from ui.main_window import MainWindow
from utils.port_manager import PortManager
import config  # config.py를 임포트한다고 가정

detected_port = PortManager.update_config("config.py")

if detected_port:
    config.SERIAL_PORT = detected_port

def setup_environment():
    """윈도우 환경 설정 (DPI 인식, 콘솔 타이틀 등)"""
    try:
        # High DPI 설정 (좌표 밀림 방지)
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
    
    # 윈도우 타이틀 설정
    try:
        ctypes.windll.kernel32.SetConsoleTitleW("MapleHunter Bot Console")
    except:
        pass

def main():
    # 1. 환경 초기화
    setup_environment()
    logger.info("============== [ MapleHunter Bot v2.0 Started ] ==============")

    try:
        # 2. 핵심 에이전트(Controller) 초기화
        # - Vision, Scanner, ActionHandler 등을 내부적으로 생성합니다.
        # - 초기에는 맵 없이 시작하며, UI에서 로드합니다.
        agent = BotAgent()
        logger.info("✅ BotAgent initialized.")

        # 3. UI(View) 초기화 및 연결
        root = tk.Tk()
        
        # 앱 실행 시 창을 잠시 맨 앞으로 가져옴
        root.attributes('-topmost', True)
        root.update()
        root.attributes('-topmost', False)
        
        # MainWindow 생성 (Agent를 주입받아 데이터에 접근)
        app = MainWindow(root, agent)
        logger.info("✅ Main Window GUI loaded.")

        # 4. 종료 처리 핸들러
        def on_closing():
            if messagebox.askokcancel("종료", "봇을 종료하시겠습니까?"):
                logger.info("Closing application...")
                if agent.running:
                    agent.stop()
                root.destroy()
                sys.exit(0)

        # Tkinter X버튼(종료) 이벤트 연결
        from tkinter import messagebox
        root.protocol("WM_DELETE_WINDOW", on_closing)

        # 5. 메인 루프 실행
        logger.info("🚀 Entering Main Loop...")
        root.mainloop()

    except Exception as e:
        logger.critical(f"Fatal Error in main execution: {e}")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("============== [ MapleHunter Bot Terminated ] ==============")

if __name__ == "__main__":
    main()