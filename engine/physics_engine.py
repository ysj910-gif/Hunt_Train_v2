# Hunt_Train_v2/engine/physics_engine.py

import torch
import os
import config
# [수정] 두 클래스 모두 임포트
from modules.physics_model import HybridPhysicsNet, LegacyPhysicsNet
from utils.logger import logger

class PhysicsEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            logger.warning(f"Physics model not found: {file_path}")
            return False
            
        try:
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
            
            # 모델 구조 자동 감지 로직
            if 'physics_params.weight' in state_dict:
                # 구버전 모델 감지
                num_actions = state_dict['physics_params.weight'].shape[0]
                logger.info(f"구버전 물리 모델 감지 (Actions: {num_actions})")
                self.model = LegacyPhysicsNet(num_actions=num_actions).to(self.device)
                
                # [수정] strict=False 추가 (gravity 누락 허용)
                self.model.load_state_dict(state_dict, strict=False)
                
            else:
                if 'velocity.weight' in state_dict:
                    # 신버전(Hybrid) 모델 감지
                    num_actions = state_dict['velocity.weight'].shape[0]
                    self.model = HybridPhysicsNet(num_actions=num_actions).to(self.device)
                    self.model.load_state_dict(state_dict)
                else:
                    # [수정] 16 -> config.PHYSICS_ACTION_DIM_DEFAULT
                    num_actions = config.PHYSICS_ACTION_DIM_DEFAULT
                
                logger.info(f"신버전 물리 모델 감지 (Actions: {num_actions})")
                self.model = HybridPhysicsNet(num_actions=num_actions).to(self.device)
                self.model.load_state_dict(state_dict)
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Physics engine loaded successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load physics model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def predict_velocity(self, action_idx, is_ground):
        """특정 행동 시 예상 속도와 중력을 반환"""
        if not self.is_loaded:
            return None
            
        try:
            with torch.no_grad():
                act_tensor = torch.LongTensor([action_idx]).to(self.device)
                ground_tensor = torch.FloatTensor([float(is_ground)]).to(self.device)
                
                vel, gravity = self.model(act_tensor, ground_tensor)
                
                # 결과 반환 (속도[vx, vy], 중력)
                return vel.cpu().numpy()[0], gravity.item()
        except Exception as e:
            logger.error(f"Physics prediction error: {e}")
            return None