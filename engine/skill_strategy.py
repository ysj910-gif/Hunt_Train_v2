# engine/skill_strategy.py

import time
from utils.logger import logger, trace_logic
from utils.physics_utils import PhysicsUtils

class SkillStrategy:
    """
    스킬 사용의 우선순위와 타이밍을 결정하는 지능 계층 모듈입니다.
    Scanner로부터의 쿨타임 정보와 PathFinder의 위치 전략을 결합합니다.
    """
    def __init__(self, path_finder):
        self.path_finder = path_finder
        self.skill_priorities = {
            "buff": 1,      # 최우선 (공격력 증가 등)
            "install": 2,   # 설치기 (Fountain 등)
            "main": 3,      # 주력 공격기
            "utility": 4    # 보조기
        }
        # 스킬별 설정 (이름, 타입, 쿨타임 등)
        self.skills = {} 

    def register_skill_info(self, name, skill_type, cooldown):
        """봇이 사용할 스킬의 메타데이터를 등록합니다."""
        self.skills[name] = {
            "type": skill_type,
            "cooldown": cooldown,
            "last_used": 0,
            "priority": self.skill_priorities.get(skill_type, 9)
        }

    @trace_logic
    def decide_skill(self, current_pos, scanner_status):
        """
        현재 상황에서 가장 적절한 스킬 사용을 결정합니다.
        """
        available_skills = []

        for name, status in scanner_status.items():
            # [수정 전] if status and name in self.skills:
            # [수정 후] status가 False(쿨타임 아님)일 때 사용 가능
            if not status and name in self.skills:
                available_skills.append(name)

        if not available_skills:
            return None

        # 1. 우선순위에 따라 정렬
        available_skills.sort(key=lambda n: self.skills[n]['priority'])

        for skill_name in available_skills:
            skill = self.skills[skill_name]

            # 2. 설치기(Install) 특수 판단 로직
            if skill['type'] == "install":
                best_spot = self.path_finder.get_best_install_spot()
                if best_spot:
                    # 현재 위치가 최적의 설치 명당과 가까운지 확인
                    dist = PhysicsUtils.calc_distance(current_pos, best_spot)
                    if dist < 20: # 명당 도착 시 사용
                        return self._confirm_decision(skill_name, "At best install spot")
                    else:
                        # 명당이 멀다면 설치를 미루고 이동 우선
                        continue

            # 3. 버프 및 주력기 판단
            if skill['type'] in ["buff", "main"]:
                return self._confirm_decision(skill_name, f"Priority {skill['type']} ready")

        return None

    def _confirm_decision(self, skill_name, reason):
        """결정을 확정하고 로그를 남깁니다."""
        self.skills[skill_name]['last_used'] = time.time()
        logger.log_decision(
            step="SkillStrategy",
            state="COMBAT",
            decision=f"Use {skill_name}",
            reason=reason
        )
        return skill_name

    def get_refill_request(self, scanner_status):
        """설치기가 만료되었거나 곧 만료될 경우 PathFinder에게 이동을 요청하기 위한 판단"""
        for name, info in self.skills.items():
            if info['type'] == "install":
                # 스캐너상 사용 가능한데(쿨타임 돌았는데) 설치된 게 없다면
                if scanner_status.get(name) and name not in self.path_finder.installed_objects:
                    return True
        return False