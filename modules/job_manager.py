import json
import os
import config
from utils.logger import logger

class JobManager:
    """
    직업별 키 매핑 및 스킬 설정을 관리하는 클래스.
    UI(SkillTab)와 봇(BotAgent) 모두에서 호환되도록 설계되었습니다.
    """
    def __init__(self, job_file="jobs.json"):
        self.job_file = job_file
        self.jobs_data = {}
        
        # [봇 설정] 현재 선택된 직업 정보
        self.current_job_name = config.CURRENT_JOB
        self.current_mapping = config.DEFAULT_KEYS.copy()
        self.skill_info = {}
        
        # [UI 설정] 직업 이름 <-> ID 매핑
        self.job_map = {} 

        self.load_jobs()

    def load_jobs(self):
        """jobs.json 파일을 로드하고 파싱합니다."""
        if not os.path.exists(self.job_file):
            logger.warning(f"⚠️ {self.job_file} not found. Creating a new one with defaults.")
            self._create_default_file()
            return

        try:
            with open(self.job_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.jobs_data = data
            
            # 1. UI용 ID 맵 생성 (순서대로 인덱스 부여)
            self.job_map = {name: idx for idx, name in enumerate(data.keys())}

            # 2. 봇용 현재 직업 설정 로드
            if self.current_job_name in self.jobs_data:
                job_content = self.jobs_data[self.current_job_name]
                
                # dict 형태인지 확인 (새로운 포맷 호환성)
                if isinstance(job_content, dict):
                    # 키 매핑 로드 (기본값 위에 덮어쓰기)
                    if "key_mapping" in job_content:
                        self.current_mapping.update(job_content["key_mapping"])
                    
                    # 스킬 정보 로드
                    if "skill_settings" in job_content:
                        self.skill_info = job_content["skill_settings"]
                else:
                    # 구버전 포맷(단순 ID)일 경우 경고
                    logger.warning(f"⚠️ Job '{self.current_job_name}' has old format. Please update jobs.json.")

                logger.info(f"✅ Loaded settings for job: {self.current_job_name}")
            else:
                logger.warning(f"⚠️ Job '{self.current_job_name}' not found in {self.job_file}. Using defaults.")

        except Exception as e:
            logger.error(f"❌ Failed to parse {self.job_file}: {e}")
            # 파싱 실패 시 빈 상태로 두어 크래시 방지
            self.job_map = {}
            self.jobs_data = {}

    def _create_default_file(self):
        """파일이 없을 때 기본 템플릿 생성"""
        default_data = {
            "Kinesis": {
                "key_mapping": config.DEFAULT_KEYS,
                "skill_settings": {
                    "fountain": {"cooldown": 60.0, "type": "install"},
                    "ultimate": {"cooldown": 120.0, "type": "buff"}
                }
            }
        }
        try:
            with open(self.job_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=4, ensure_ascii=False)
            self.load_jobs() # 생성 후 다시 로드
        except Exception as e:
            logger.error(f"Failed to create default jobs.json: {e}")

    def _save_jobs(self):
        """현재 직업 데이터(self.jobs_data)를 파일에 저장"""
        try:
            with open(self.job_file, 'w', encoding='utf-8') as f:
                json.dump(self.jobs_data, f, ensure_ascii=False, indent=4)
            logger.info("💾 Jobs saved successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to save jobs file: {e}")

    # ==============================
    # [UI 호환 메서드]
    # ==============================
    def get_all_jobs(self):
        """등록된 모든 직업 이름 리스트 반환"""
        return list(self.jobs_data.keys())

    def get_job_id(self, job_name):
        """
        직업 이름에 해당하는 ID를 반환.
        등록되지 않은 직업이면 구조를 초기화하여 등록 후 저장.
        """
        if job_name not in self.job_map:
            # 새로운 직업 추가 (기본 템플릿으로)
            new_id = len(self.jobs_data)
            self.jobs_data[job_name] = {
                "key_mapping": config.DEFAULT_KEYS.copy(),
                "skill_settings": {}
            }
            self.job_map[job_name] = new_id
            
            logger.info(f"🆕 Registered new job: {job_name} (ID: {new_id})")
            self._save_jobs()
            
        return self.job_map[job_name]

    # ==============================
    # [봇 로직 메서드]
    # ==============================
    def get_key_mapping(self):
        """현재 직업의 키 매핑 반환"""
        return self.current_mapping

    def get_skill_cooldown(self, skill_name):
        """특정 스킬의 쿨타임 반환"""
        return self.skill_info.get(skill_name, {}).get("cooldown", 0.0)
    
    def get_skill_type(self, skill_name):
        """특정 스킬의 타입 반환 (install, buff 등)"""
        return self.skill_info.get(skill_name, {}).get("type", "unknown")