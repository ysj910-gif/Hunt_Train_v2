# ui/simulation_mode.py

import tkinter as tk
import time
from engine.map_processor import MapProcessor
from engine.path_finder import PathFinder
from engine.physics_engine import PhysicsEngine
from engine.skill_strategy import SkillStrategy
from utils.logger import logger, trace_logic
from ui.components.viewport_manager import ViewportManager
import math

class SimulationMode:
    """
    시뮬레이션 모드의 로직과 렌더링을 담당하는 클래스
    메인 윈도우의 캔버스를 공유하여 사용합니다.
    """
    def __init__(self, main_window):
        self.mw = main_window
        self.canvas = main_window.canvas
        self.root = self.canvas.winfo_toplevel()
        
        # 뷰포트 관리자
        self.viewport = ViewportManager() 
        
        # 1. 엔진 초기화
        self.map_processor = self.mw.agent.map_processor
        self.physics_engine = PhysicsEngine()
        
        if self.mw.cur_rf_path:
            self.physics_engine.load_model(self.mw.cur_rf_path)
        
        self.path_finder = PathFinder(self.map_processor, self.physics_engine)
        self.skill_strategy = SkillStrategy(self.path_finder)
        
        # 2. 캐릭터 상태 변수
        self.char_x = 125.0
        self.char_y = 66.0
        self.vx = 0.0
        self.vy = 0.0
        self.is_ground = True
        
        self.active = False
        self.last_time = time.time()
        
        # 3. 입력 및 스킬 시뮬레이션 변수
        self.pressed_keys = set()
        self.bind_ids = []
        
        self.sim_key_map = {}   # { 'z': 'Fountain', ... } (키 -> 스킬이름 매핑)
        self.sim_skills = {}    # { 'Fountain': {duration, cooldown, ...} }
        self.sim_cooldowns = {} # { 'Fountain': ready_time }
        self.installed_objects = [] # [{x, y, name, expire_time}, ...]
        
        # 맵 데이터에 맞춰 월드 크기 설정
        if self.map_processor.platforms:
            max_x = max(p['x_end'] for p in self.map_processor.platforms)
            max_y = max(p['y'] for p in self.map_processor.platforms)
            self.viewport.set_world_size(max_x + 20, max_y + 20)
        else:
            self.viewport.set_world_size(300, 200)
            
        self.viewport.zoom_scale = 4.0 
        self.viewport.center_view()

    def start(self):
        self.active = True
        self.last_time = time.time()
        
        # 상태 리셋
        self.char_x = 125.0
        self.char_y = 66.0
        self.vx, self.vy = 0.0, 0.0
        self.pressed_keys.clear()
        self.installed_objects.clear()
        self.sim_cooldowns.clear()
        
        # [핵심] Skill Tab에서 정보 가져오기
        self.import_skills_from_tab()
        
        # 입력 바인딩 활성화
        self.bind_inputs()
        
        print(f">>> 시뮬레이션 모드 시작")
        print(f"    - 조작: 방향키(이동), Alt(점프)")
        print(f"    - 스킬: {list(self.sim_key_map.keys())} 키로 테스트 가능")

    def stop(self):
        self.active = False
        self.unbind_inputs()
        print(">>> 시뮬레이션 모드 종료")

    def import_skills_from_tab(self):
        """SkillTab에 입력된 내용을 시뮬레이터로 로드합니다."""
        self.sim_key_map.clear()
        self.sim_skills.clear()
        
        if not self.mw.skill_tab:
            return

        # 1. 설치기 (Install Rows) 가져오기
        # SkillTab의 add_install_row 구조: {'name': Entry, 'key': Entry, 'range': Entry, 'dur': Entry, ...}
        for row in self.mw.skill_tab.install_rows:
            try:
                # 위젯이 존재하는지 확인
                if not row['frame'].winfo_exists(): continue
                
                name = row['name'].get().strip()
                key_str = row['key'].get().strip().lower() # 소문자로 통일
                dur_str = row['dur'].get().strip()
                
                if not name or not key_str: continue
                
                duration = float(dur_str) if dur_str else 10.0
                
                # 등록
                self.sim_key_map[key_str] = name
                self.sim_skills[name] = {
                    "type": "install",
                    "duration": duration,
                    "cooldown": 3.0 # 설치기 쿨타임은 탭에 없으므로 기본 3초 부여
                }
            except Exception as e:
                print(f"[Sim] 스킬 로드 실패: {e}")

    def bind_inputs(self):
        self.bind_ids.append(self.root.bind("<KeyPress>", self.on_key_press))
        self.bind_ids.append(self.root.bind("<KeyRelease>", self.on_key_release))
        self.canvas.focus_set()

    def unbind_inputs(self):
        for bind_id in self.bind_ids:
            self.root.unbind("<KeyPress>", bind_id)
            self.root.unbind("<KeyRelease>", bind_id)
        self.bind_ids = []

    def on_key_press(self, event):
        if not self.active: return
        
        key = event.keysym
        char_key = event.char.lower() # 문자 키 (a, z, etc.)
        
        self.pressed_keys.add(key)
        
        # 1. 점프 (Alt)
        if key in ["Alt_L", "Alt_R"]:
            if "Up" in self.pressed_keys:
                self.apply_action("up_jump")
            else:
                self.apply_action("jump")
        
        # 2. 스킬 단축키 확인 (char_key 사용)
        # Tkinter의 event.char는 한글 입력 상태 등에 따라 비어있을 수 있으므로 주의
        check_key = char_key if char_key else key.lower()
        
        if check_key in self.sim_key_map:
            skill_name = self.sim_key_map[check_key]
            self.try_use_skill(skill_name)

    def on_key_release(self, event):
        if event.keysym in self.pressed_keys:
            self.pressed_keys.remove(event.keysym)

    @trace_logic
    def update(self):
        if not self.active: return

        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        # 0. 이동 입력 처리 (지속 입력)
        if "Left" in self.pressed_keys:
            self.apply_action("move_left")
        elif "Right" in self.pressed_keys:
            self.apply_action("move_right")
        
        # 1. 물리 업데이트
        gravity = 5.0 
        if not self.is_ground:
            self.vy += gravity * 0.1
            
        self.char_x += self.vx
        self.char_y += self.vy
        
        # 2. 지형 충돌
        plat = self.map_processor.find_current_platform(self.char_x, self.char_y)
        if plat:
            if self.vy > 0 and abs(self.char_y - plat['y']) < 6:
                self.char_y = plat['y']
                self.vy = 0
                self.is_ground = True
        else:
            self.is_ground = False
            
        if self.char_y > 300: # 낙하 리셋
            self.char_y = 66
            self.vy = 0
            self.is_ground = True

        if self.is_ground:
            self.vx *= 0.8
            if abs(self.vx) < 0.1: self.vx = 0
            
        # [설치물] 지속시간 만료 처리
        current_time = time.time()
        self.installed_objects = [obj for obj in self.installed_objects if obj['expire_time'] > current_time]

        # 3. 그리기
        self.draw()

    def try_use_skill(self, skill_name):
        info = self.sim_skills.get(skill_name)
        if not info: return

        current_time = time.time()
        
        # 쿨타임 체크
        ready_time = self.sim_cooldowns.get(skill_name, 0)
        if current_time < ready_time:
            left = ready_time - current_time
            print(f"⏳ {skill_name} 쿨타임! ({left:.1f}s 남음)")
            return

        # 설치기 로직
        if info['type'] == "install":
            if not self.is_ground:
                print("🚫 설치기는 지상에서만 사용 가능합니다.")
                return
                
            self.installed_objects.append({
                "name": skill_name,
                "x": self.char_x,
                "y": self.char_y,
                "expire_time": current_time + info['duration']
            })
            
            # 쿨타임 적용
            self.sim_cooldowns[skill_name] = current_time + info['cooldown']
            print(f"✅ [설치] {skill_name} 설치됨 (지속 {info['duration']}s)")

    @trace_logic
    def apply_action(self, action_name):
        """사용자 입력 또는 봇의 행동을 물리 엔진에 반영"""
        # [디버깅] 입력된 액션 이름 확인
        print(f"[DEBUG] Action Input: {action_name}")
        if not self.physics_engine.is_loaded:
            if action_name == "move_left": self.vx = -3
            elif action_name == "move_right": self.vx = 3
            elif action_name == "jump" and self.is_ground: self.vy = -5; self.is_ground = False
            elif action_name == "up_jump": self.vy = -12; self.is_ground = False
            return

        act_map = {"move_left": 0, "move_right": 1, "jump": 2, "up_jump": 3}
        idx = act_map.get(action_name, -1)
        
        if idx != -1:
            if "jump" in action_name and not self.is_ground: return
            
            vel, gravity = self.physics_engine.predict_velocity(idx, self.is_ground)
            
            # [디버깅] 물리 엔진이 예측한 속도 값 확인
            # move_right라면 vel[0]이 양수(+)여야 하고, move_left라면 음수(-)여야 합니다.
            print(f"[DEBUG] Model Output for {action_name} (idx {idx}): vx={vel[0]:.4f}, vy={vel[1]:.4f}")

            if "move" in action_name: 
                self.vx = float(vel[0])
            else: 
                self.vx, self.vy = float(vel[0]), float(vel[1])
                self.is_ground = False

    @trace_logic
    def draw(self):
        self.canvas.delete("sim_obj") 
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        def to_screen(x, y):
            return self.viewport.world_to_screen(x, y, cw, ch)

        # 1. 발판
        for plat in self.map_processor.platforms:
            x1, y1 = to_screen(plat['x_start'], plat['y'])
            x2, y2 = to_screen(plat['x_end'], plat['y'])
            self.canvas.create_line(x1, y1, x2, y2, fill="#00FF00", width=3, tags="sim_obj")

        # 2. 설치물 (정보 표시 포함)
        for obj in self.installed_objects:
            ox, oy = to_screen(obj['x'], obj['y'])
            # 파란색 기둥 느낌
            self.canvas.create_rectangle(ox-10, oy-40, ox+10, oy, fill="#3333FF", outline="cyan", tags="sim_obj")
            
            # 남은 시간 표시
            left_time = max(0, obj['expire_time'] - time.time())
            self.canvas.create_text(ox, oy-50, text=f"{obj['name']}\n{left_time:.1f}s", 
                                  fill="cyan", font=("Arial", 8), justify="center", tags="sim_obj")

        # 3. 캐릭터
        px, py = to_screen(self.char_x, self.char_y)
        self.canvas.create_oval(px-10, py-25, px+10, py, fill="#FF5555", outline="white", width=2, tags="sim_obj")
        
        # 4. 정보
        info = f"Pos: ({self.char_x:.1f}, {self.char_y:.1f})"
        if self.sim_key_map:
            info += f"\nActive Skills: {', '.join([f'{k.upper()}:{n}' for k, n in self.sim_key_map.items()])}"
            
        self.canvas.create_text(10, 10, anchor="nw", text=info, fill="yellow", font=("Arial", 10), tags="sim_obj")