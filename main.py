# main.py

import sys
import os
import tkinter as tk
import ctypes
import traceback
from tkinter import messagebox

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Config & Utils
import config
from utils.logger import logger
from utils.port_manager import PortManager

# Modules (부품들)
from modules.vision_system import VisionSystem
from modules.scanner import GameScanner
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from engine.physics_engine import PhysicsEngine
from core.action_handler import ActionHandler
from core.data_recorder import DataRecorder

# Core (본체 및 두뇌)
from core.bot_agent import BotAgent
from core.decision_maker import DecisionMaker

# UI
from ui.main_window import MainWindow

def setup_windows_environment():
    """DPI 및 콘솔 설정"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
    
    try:
        ctypes.windll.kernel32.SetConsoleTitleW("MapleHunter Bot Console")
    except:
        pass

def main():
    setup_windows_environment()
    logger.info("============== [ MapleHunter Bot v2.0 ] ==============")

    try:
        # =========================================================
        # 1. Dependency Construction (부품 생성)
        # =========================================================
        
        # 1-1. 하드웨어/포트 설정
        detected_port = PortManager.update_config("config.py")
        if detected_port:
            config.SERIAL_PORT = detected_port
            
        action_mode = "HARDWARE" if getattr(config, 'SERIAL_PORT', None) else "SOFTWARE"
        logger.info(f"⚙️ Action Mode: {action_mode} (Port: {getattr(config, 'SERIAL_PORT', 'None')})")

        # 1-2. 핵심 모듈 생성
        vision_system = VisionSystem()
        game_scanner = GameScanner()
        action_handler = ActionHandler(mode=action_mode, serial_port=getattr(config, 'SERIAL_PORT', None))
        
        map_processor = MapProcessor()
        physics_engine = PhysicsEngine()
        physics_engine.load_model("physics_hybrid_model.pth") # 모델 로드
        
        # PathFinder는 Map과 Physics가 필요
        path_finder = PathFinder(map_processor, physics_engine)
        
        # DataRecorder (선택적)
        recorder = DataRecorder("Record_Init") # 필요 시 생성

        logger.info("✅ All modules instantiated.")

        # =========================================================
        # 2. Assembly (조립)
        # =========================================================

        # 2-1. BotAgent 조립 (신체 구성)
        agent = BotAgent(
            vision=vision_system,
            scanner=game_scanner,
            action_handler=action_handler,
            map_processor=map_processor,
            path_finder=path_finder,
            recorder=recorder # 필요하면 recorder 객체 주입
        )
        
        # 키 맵핑 로드 (Config에서 읽어오거나 기본값 설정)
        agent.key_mapping = {
            'jump': 'alt',
            'main': 'delete', # 예시
            'fountain': '4',
            'ultimate': '6'
        }

        # 2-2. DecisionMaker 조립 (두뇌 장착)
        # Brain은 Agent의 상태를 읽어야 하므로 agent를 인자로 받음
        brain = DecisionMaker(agent)
        
        # Agent에게 Brain을 장착 (Setter Injection)
        agent.set_brain(brain)

        logger.info("🤖 BotAgent assembly complete.")

        # =========================================================
        # 3. UI Initialization & Run
        # =========================================================
        root = tk.Tk()
        
        # 창 맨 앞으로 가져오기 트릭
        root.attributes('-topmost', True)
        root.update()
        root.attributes('-topmost', False)
        
        # MainWindow에 완성된 Agent 주입
        app = MainWindow(root, agent)
        logger.info("🖥️ UI Loaded.")

        # 종료 핸들러
        def on_closing():
            if messagebox.askokcancel("종료", "봇을 종료하시겠습니까?"):
                logger.info("Closing application...")
                if agent.running:
                    agent.stop()
                root.destroy()
                sys.exit(0)

        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        logger.info("🚀 System Ready. Entering Main Loop...")
        root.mainloop()

    except Exception as e:
        logger.critical(f"🔥 Fatal Error in main assembly: {e}")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("============== [ Terminated ] ==============")

if __name__ == "__main__":
    main()