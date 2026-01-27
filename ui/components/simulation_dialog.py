import sys
import os
import math
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QWidget, QSpinBox, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

# 엔진 모듈 임포트
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from utils.logger import logger, trace_logic

class SimulationCanvas(QWidget):
    """맵과 캐릭터를 그리는 캔버스 위젯"""
    def __init__(self, map_processor, parent=None):
        super().__init__(parent)
        self.map_processor = map_processor
        self.char_pos = (0, 0)
        self.target_pos = None
        self.path = []
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: #333333;") # 어두운 배경

    def update_state(self, char_pos, target_pos=None, path=None):
        self.char_pos = char_pos
        self.target_pos = target_pos
        self.path = path if path else []
        self.update() # 다시 그리기 요청

    @trace_logic
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 좌표 변환 (맵 좌표 -> 화면 좌표)
        # 미니맵이 작으므로 화면에 꽉 차게 스케일링
        scale = 3.0 
        offset_x = 50
        offset_y = 50

        def to_screen(x, y):
            return x * scale + offset_x, y * scale + offset_y

        # 2. 발판 그리기
        painter.setPen(QPen(QColor("#00FF00"), 2))
        for plat in self.map_processor.platforms:
            x1, y1 = to_screen(plat['x_start'], plat['y'])
            x2, y2 = to_screen(plat['x_end'], plat['y'])
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            # 발판 ID 표시
            # painter.drawText(int((x1+x2)/2), int(y1)-5, str(plat.get('id', '')))

        # 3. 목표 지점 그리기
        if self.target_pos:
            tx, ty = to_screen(*self.target_pos)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 0, 150)) # 반투명 노랑
            painter.drawEllipse(int(tx)-5, int(ty)-5, 10, 10)

        # 4. 경로 그리기
        # (구현 가능: Path 노드들을 선으로 연결)

        # 5. 캐릭터 그리기
        cx, cy = to_screen(*self.char_pos)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(QColor("#FF5555"))
        painter.drawEllipse(int(cx)-6, int(cy)-15, 12, 15) # 단순화된 캐릭터

class SimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Mode 🧪")
        self.resize(900, 600)
        
        # 1. 엔진 초기화
        self.map_processor = MapProcessor()
        # 맵 파일 로드 (경로 확인 필요)
        map_path = "Corrected_Royal_Library_6.json"
        if os.path.exists(map_path):
            self.map_processor.load_map(map_path)
        else:
            print(f"맵 파일을 찾을 수 없습니다: {map_path}")

        # 물리 엔진 없이 로직 테스트 (A* 내부 시뮬레이션 사용)
        self.path_finder = PathFinder(self.map_processor, physics_engine=None)
        
        # 초기 상태
        self.char_x = 125
        self.char_y = 66
        self.install_ready = {"fountain": True}
        
        # UI 구성
        self.init_ui()
        
        # 봇 설정 등록 (테스트용)
        self.path_finder.register_install_skill("fountain", {'left': 100, 'right': 100, 'up': 50, 'down': 50}, 60.0)

    def init_ui(self):
        layout = QHBoxLayout()

        # [왼쪽] 캔버스 영역
        self.canvas = SimulationCanvas(self.map_processor)
        layout.addWidget(self.canvas, stretch=2)

        # [오른쪽] 제어 패널
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()

        # 1. 위치 설정
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Start X:"))
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 300)
        self.spin_x.setValue(self.char_x)
        pos_layout.addWidget(self.spin_x)
        
        pos_layout.addWidget(QLabel("Start Y:"))
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 300)
        self.spin_y.setValue(self.char_y)
        pos_layout.addWidget(self.spin_y)
        
        btn_place = QPushButton("Place Character")
        btn_place.clicked.connect(self.on_place_character)
        pos_layout.addWidget(btn_place)
        
        control_layout.addLayout(pos_layout)

        # 2. 수동 조작 시뮬레이션
        input_group = QGroupBox("Manual Input Simulation")
        input_layout = QHBoxLayout()
        
        btn_left = QPushButton("⬅️ Walk Left")
        btn_left.clicked.connect(lambda: self.simulate_move(-2, 0)) # 단순 이동
        input_layout.addWidget(btn_left)

        btn_right = QPushButton("➡️ Walk Right")
        btn_right.clicked.connect(lambda: self.simulate_move(2, 0))
        input_layout.addWidget(btn_right)
        
        btn_jump = QPushButton("🦘 Jump")
        btn_jump.clicked.connect(lambda: self.simulate_move(0, -10)) # 단순 점프
        input_layout.addWidget(btn_jump)

        input_group.setLayout(input_layout)
        control_layout.addWidget(input_group)

        # 3. 봇 로직 테스트
        bot_group = QGroupBox("Bot Logic Test")
        bot_layout = QVBoxLayout()
        
        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: yellow; font-weight: bold;")
        bot_layout.addWidget(self.lbl_status)

        btn_bot_step = QPushButton("🤖 Run Next Bot Step")
        btn_bot_step.clicked.connect(self.run_bot_logic)
        bot_layout.addWidget(btn_bot_step)
        
        # 쿨타임 리셋 버튼
        btn_reset = QPushButton("Reset Skill Cooldowns")
        btn_reset.clicked.connect(self.reset_logic)
        bot_layout.addWidget(btn_reset)

        bot_group.setLayout(bot_layout)
        control_layout.addWidget(bot_group)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel, stretch=1)

        self.setLayout(layout)
        
        # 초기 화면 갱신
        self.canvas.update_state((self.char_x, self.char_y))

    def on_place_character(self):
        """좌표 입력값으로 캐릭터 위치 강제 이동"""
        self.char_x = self.spin_x.value()
        self.char_y = self.spin_y.value()
        self.canvas.update_state((self.char_x, self.char_y))
        logger.info(f"Character placed at ({self.char_x}, {self.char_y})")

    def simulate_move(self, dx, dy):
        """수동 입력 시뮬레이션 (단순 좌표 이동 + 중력 처리)"""
        # 실제 물리 엔진을 연동하려면 여기서 PhysicsEngine.predict() 등을 호출
        # 여기서는 단순 시각화를 위해 좌표만 변경
        self.char_x += dx
        self.char_y += dy
        
        # 바닥 충돌 체크 (단순화)
        plat = self.map_processor.find_current_platform(self.char_x, self.char_y)
        if plat:
            self.char_y = plat['y'] # 착지
            
        self.spin_x.setValue(int(self.char_x))
        self.spin_y.setValue(int(self.char_y))
        self.canvas.update_state((self.char_x, self.char_y))

    def run_bot_logic(self):
        """봇의 판단 로직을 한 단계 실행"""
        current_pos = (self.char_x, self.char_y)
        
        # PathFinder에게 다음 행동 물어보기
        command, target = self.path_finder.get_next_combat_step(current_pos, self.install_ready)
        
        # 결과 표시
        status_text = f"Action: {command}\nTarget: {target}"
        if self.path_finder.locked_install_target:
            status_text += "\n[Locked Target]"
            
        self.lbl_status.setText(status_text)
        
        # 시각적 업데이트 (목표 지점 표시)
        actual_target = target if isinstance(target, tuple) else None
        # target이 ('execute_path', 'up_jump') 같은 튜플일 수 있으므로 처리
        if isinstance(target, str): actual_target = None
        
        self.canvas.update_state(current_pos, target_pos=actual_target)
        
        # (선택 사항) 봇의 판단대로 캐릭터를 실제로 이동시켜보고 싶다면:
        # if command == "move_to_install": ...

    def reset_logic(self):
        self.path_finder.locked_install_target = None
        self.path_finder.installed_objects = []
        self.lbl_status.setText("Status: Reset Done")
        logger.info("Bot logic reset.")