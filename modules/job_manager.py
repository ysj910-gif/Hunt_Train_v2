import json
import os
import config
from utils.logger import logger

class JobManager:
    """
    직업 이름과 ID 매핑을 관리하는 클래스.
    새로운 직업이 발견되면 자동으로 ID를 부여하고 jobs.json에 저장합니다.
    """
    def __init__(self, job_file="jobs.json"):
        self.job_file = job_file
        self.jobs_data = {}
        self.current_job_name = config.CURRENT_JOB
        self.current_mapping = config.DEFAULT_KEYS.copy()
        self.skill_info = {}
        
        self.load_jobs()

    def load_jobs(self):
        if not os.path.exists(self.job_file):
            logger.warning(f"⚠️ {self.job_file} not found. Using default keys.")
            return

        try:
            with open(self.job_file, 'r', encoding='utf-8') as f:
                self.jobs_data = json.load(f)
                
            if self.current_job_name in self.jobs_data:
                job_data = self.jobs_data[self.current_job_name]
                
                # 키 매핑 로드 (기본값 위에 덮어쓰기)
                if "key_mapping" in job_data:
                    self.current_mapping.update(job_data["key_mapping"])
                    
                # 스킬 정보 로드 (쿨타임 등)
                if "skill_settings" in job_data:
                    self.skill_info = job_data["skill_settings"]
                    
                logger.info(f"✅ Loaded settings for job: {self.current_job_name}")
            else:
                logger.warning(f"⚠️ Job '{self.current_job_name}' not found in {self.job_file}. Using defaults.")
                
        except Exception as e:
            logger.error(f"❌ Failed to parse {self.job_file}: {e}")

    def get_key_mapping(self):
        return self.current_mapping

    def get_skill_cooldown(self, skill_name):
        return self.skill_info.get(skill_name, {}).get("cooldown", 0)

    def _save_jobs(self):
        """직업 목록 저장"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.job_map, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"직업 파일 저장 실패: {e}")

    def get_job_id(self, job_name):
        """
        직업 이름에 해당하는 ID를 반환.
        등록되지 않은 직업이면 새 ID를 발급하고 저장.
        """
        if job_name not in self.job_map:
            new_id = len(self.job_map)
            self.job_map[job_name] = new_id
            self._save_jobs()
            logger.info(f"새로운 직업 등록: {job_name} (ID: {new_id})")
            
        return self.job_map[job_name]

    def get_all_jobs(self):
        """GUI 콤보박스용 직업 이름 리스트 반환"""
        return list(self.job_map.keys())