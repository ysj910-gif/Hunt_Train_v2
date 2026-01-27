import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bot_agent import BotAgent
from utils.logger import logger

class PhysicsTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍁 Maple Physics Trainer (Data Collector)")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        
        # 스타일 설정
        self.style = ttk.Style()
        self.style.configure("Big.TLabel", font=("Helvetica", 16, "bold"), foreground="#333")
        self.style.configure("Instruction.TLabel", font=("Helvetica", 14), foreground="blue")
        self.style.configure("Status.TLabel", font=("Arial", 10), foreground="gray")
        
        # 변수 초기화
        self.agent = None
        self.is_running = False
        self.map_path = tk.StringVar()
        self.current_instruction = tk.StringVar(value="맵 파일을 로드하고 훈련을 시작하세요.")
        self.progress_var = tk.DoubleVar(value=0)
        
        # UI 구성
        self._create_widgets()
        
        # 종료 시 처리
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        # 1. 맵 파일 선택 영역
        frame_top = ttk.LabelFrame(self.root, text="Step 1: 맵 데이터 로드", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)
        
        entry_map = ttk.Entry(frame_top, textvariable=self.map_path, width=40)
        entry_map.pack(side="left", padx=5, fill="x", expand=True)
        
        btn_browse = ttk.Button(frame_top, text="찾아보기...", command=self._browse_map)
        btn_browse.pack(side="right")

        # 2. 미션 리스트 (Treeview)
        frame_list = ttk.LabelFrame(self.root, text="Step 2: 훈련 커리큘럼", padding=10)
        frame_list.pack(fill="both", expand=True, padx=10, pady=5)
        
        cols = ("Step", "Mission", "Status")
        self.tree = ttk.Treeview(frame_list, columns=cols, show="headings", height=10)
        self.tree.heading("Step", text="No.")
        self.tree.heading("Mission", text="미션 내용")
        self.tree.heading("Status", text="상태")
        
        self.tree.column("Step", width=40, anchor="center")
        self.tree.column("Mission", width=250)
        self.tree.column("Status", width=80, anchor="center")
        
        # 기본 미션 목록 등록
        self.missions = [
            ("1", "가장 긴 발판으로 이동", "대기"),
            ("2", "마찰력 테스트 (오른쪽)", "대기"),
            ("3", "마찰력 테스트 (왼쪽)", "대기"),
            ("4", "가장 높은 곳으로 이동", "대기"),
            ("5", "중력 테스트 (제자리 점프)", "대기"),
            ("6", "관성 테스트 (더블 점프)", "대기")
        ]
        for item in self.missions:
            self.tree.insert("", "end", values=item)
            
        self.tree.pack(fill="both", expand=True)

        # 3. 지시 사항 및 상태 패널
        frame_status = ttk.LabelFrame(self.root, text="Step 3: 트레이너 지시사항", padding=15)
        frame_status.pack(fill="x", padx=10, pady=5)
        
        lbl_inst = ttk.Label(frame_status, textvariable=self.current_instruction, style="Instruction.TLabel", wraplength=450, anchor="center")
        lbl_inst.pack(pady=10)
        
        self.progress = ttk.Progressbar(frame_status, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", pady=5)

        # 4. 제어 버튼
        frame_ctrl = ttk.Frame(self.root, padding=10)
        frame_ctrl.pack(fill="x")
        
        self.btn_start = ttk.Button(frame_ctrl, text="🚀 훈련 시작", command=self._start_training, state="disabled")
        self.btn_start.pack(side="left", fill="x", expand=True, padx=5)
        
        btn_stop = ttk.Button(frame_ctrl, text="중지 및 저장", command=self._stop_training)
        btn_stop.pack(side="right", fill="x", expand=True, padx=5)

    def _browse_map(self):
        file_selected = filedialog.askopenfilename(
            filetypes=[("JSON Map Files", "*.json"), ("All Files", "*.*")]
        )
        if file_selected:
            self.map_path.set(file_selected)
            self.btn_start.config(state="normal")
            self.current_instruction.set("준비 완료! '훈련 시작' 버튼을 누르세요.")

    def _update_status(self, step_idx, status):
        """Treeview의 상태 업데이트"""
        child_id = self.tree.get_children()[step_idx]
        self.tree.set(child_id, "Status", status)
        
        # 선택 포커스 이동
        self.tree.selection_set(child_id)
        self.tree.see(child_id)

    def _set_instruction(self, text, progress=0):
        self.current_instruction.set(text)
        self.progress_var.set(progress)

    def _on_close(self):
        self._stop_training()
        self.root.destroy()
        sys.exit(0)

    # --- 트레이닝 로직 ---
    def _start_training(self):
        if not self.map_path.get():
            messagebox.showwarning("경고", "먼저 맵 파일을 선택해주세요.")
            return
            
        self.is_running = True
        self.btn_start.config(state="disabled")
        
        # 별도 스레드에서 로직 실행 (GUI 멈춤 방지)
        t = threading.Thread(target=self._training_thread)
        t.daemon = True
        t.start()

    def _stop_training(self):
        self.is_running = False
        if self.agent:
            self.agent.stop()
        self.current_instruction.set("훈련이 중지되었습니다.")
        self.btn_start.config(state="normal")

    def _training_thread(self):
        try:
            # 1. 에이전트 초기화
            self.agent = BotAgent()
            if not self.agent.map_processor.load_map(self.map_path.get()):
                messagebox.showerror("에러", "맵 파일을 불러올 수 없습니다.")
                return

            # 에이전트 구동 (화면 인식 시작)
            t_agent = threading.Thread(target=self.agent.run)
            t_agent.daemon = True
            t_agent.start()
            
            # 맵 분석
            platforms = self.agent.map_processor.platforms
            run_spot = max(platforms, key=lambda p: p['x_end'] - p['x_start'])
            jump_spot = min(platforms, key=lambda p: p['y'])
            
            # --- 미션 루프 시작 ---
            
            # [Mission 1] 이동 (Run Spot)
            self._update_status(0, "진행 중...")
            self._wait_for_location(run_spot, "가장 긴 발판")
            self._update_status(0, "✅ 완료")

            # [Mission 2] 마찰력 (우)
            self._update_status(1, "진행 중...")
            self._perform_run_test("Right", "오른쪽")
            self._update_status(1, "✅ 완료")
            
            # [Mission 3] 마찰력 (좌)
            self._update_status(2, "진행 중...")
            self._perform_run_test("Left", "왼쪽")
            self._update_status(2, "✅ 완료")

            # [Mission 4] 이동 (Jump Spot)
            if run_spot != jump_spot:
                self._update_status(3, "진행 중...")
                self._wait_for_location(jump_spot, "가장 높은 발판")
            self._update_status(3, "✅ 완료")

            # [Mission 5] 중력 테스트
            self._update_status(4, "진행 중...")
            self._perform_jump_test()
            self._update_status(4, "✅ 완료")
            
            # [Mission 6] 관성 테스트
            self._update_status(5, "진행 중...")
            self._perform_inertia_test()
            self._update_status(5, "✅ 완료")

            messagebox.showinfo("완료", "모든 훈련이 끝났습니다! 데이터가 저장되었습니다.")
            self._stop_training()

        except Exception as e:
            logger.error(f"Training Error: {e}")
            self.current_instruction.set(f"오류 발생: {e}")

    # --- 세부 동작 로직 ---
    def _wait_for_location(self, target, spot_name):
        if self.agent: 
            self.agent.recorder.set_scenario("Moving")
        t_y = target['y']
        while self.is_running:
            pos = self.agent.decision_maker.scanner.player_pos
            if pos:
                px, py = pos
                if (target['x_start'] - 50 <= px <= target['x_end'] + 50) and (abs(py - t_y) <= 15):
                    break
            
            msg = f"이동하세요: {spot_name}\n좌표: Y={t_y} 근처"
            self.root.after(0, self._set_instruction, msg, 0)
            time.sleep(0.5)

    def _perform_run_test(self, direction_en, direction_kr):
        # 카운트다운
        for i in range(3, 0, -1):
            self.root.after(0, self._set_instruction, f"{direction_kr}쪽 달리기 준비... {i}", 0)
            time.sleep(1)
            
        # 달리기 지시
        self.agent.recorder.set_scenario(f"Trainer_Friction_{direction_en}")
        for i in range(30): # 3초 (0.1s * 30)
            if not self.is_running: return
            prog = (i / 30) * 100
            self.root.after(0, self._set_instruction, f"🏃 {direction_kr}쪽으로 달리세요! (유지)", prog)
            time.sleep(0.1)
            
        # [수정] 멈춤 지시 및 대기
        self.root.after(0, self._set_instruction, "🛑 키를 놓으세요! (관성 기록 중...)", 100)
        
        # 여기서 2초를 그냥 기다리는 게 아니라, 실제로 멈출 때까지 기다림
        self._wait_until_stopped() 
        
        # 다 멈춘 뒤에 시나리오 종료 (Moving으로 변경 등)
        self.agent.recorder.set_scenario("Moving")

    def _perform_jump_test(self):
        self.root.after(0, self._set_instruction, "제자리 점프 3회 (방향키 X)", 0)
        time.sleep(2)
        
        for i in range(3):
            if not self.is_running: return
            self.agent.recorder.set_scenario(f"Trainer_Jump_Neutral_{i}")
            self.root.after(0, self._set_instruction, f"🦘 점프하세요! ({i+1}/3)", (i+1)/3*100)
            time.sleep(1.5)

    def _perform_inertia_test(self):
        for i in range(3, 0, -1):
            self.root.after(0, self._set_instruction, f"더블 점프 준비... {i}", 0)
            time.sleep(1)
            
        self.agent.recorder.set_scenario("Trainer_DoubleJump")
        self.root.after(0, self._set_instruction, "🚀 달리면서 더블 점프 하세요!", 100)
        time.sleep(3)

    def _wait_until_stopped(self, timeout=5.0):
        """캐릭터가 완전히 멈출 때까지 대기 (최대 timeout초)"""
        start_time = time.time()
        stable_count = 0
        last_pos = None

        while time.time() - start_time < timeout:
            current_pos = self.agent.decision_maker.scanner.player_pos
            if not current_pos: continue
            
            # 이전 위치와 현재 위치가 거의 같으면 (오차 1~2픽셀)
            if last_pos and abs(current_pos[0] - last_pos[0]) <= 2 and abs(current_pos[1] - last_pos[1]) <= 2:
                stable_count += 1
            else:
                stable_count = 0 # 다시 움직이면 리셋
            
            last_pos = current_pos
            
            # 약 0.5초(5번 체크) 동안 움직임이 없으면 '정지'로 판정
            if stable_count >= 5:
                break
                
            time.sleep(0.1)

if __name__ == "__main__":
    root = tk.Tk()
    app = PhysicsTrainerApp(root)
    root.mainloop()