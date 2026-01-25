import json
import os
from utils.logger import logger

class JobManager:
    """
    직업 이름과 ID 매핑을 관리하는 클래스.
    새로운 직업이 발견되면 자동으로 ID를 부여하고 jobs.json에 저장합니다.
    """
    def __init__(self, filepath="jobs.json"):
        self.filepath = filepath
        self.job_map = self._load_jobs()

    def _load_jobs(self):
        """jobs.json 파일 로드"""
        if not os.path.exists(self.filepath):
            logger.info(f"직업 파일이 없어 새로 생성합니다: {self.filepath}")
            return {}
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"직업 목록 로드 완료 ({len(data)}개)")
                return data
        except Exception as e:
            logger.error(f"직업 파일 로드 실패: {e}")
            return {}

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