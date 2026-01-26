#core\model_loader.py

import torch
import numpy as np
import pandas as pd
from modules.model import LSTMModel, GRUModel
from utils.logger import logger

class ModelLoader:
    """
    PyTorch LSTM 모델을 로드하고 추론을 수행하는 래퍼 클래스.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_cols = []
        self.seq_length = 10
        self.is_loaded = False

    def load_model(self, file_path):
        if not file_path:
            return False
            
        try:
            # weights_only=False 설정 확인
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            self.scaler = checkpoint.get('scaler')
            self.encoder = checkpoint.get('encoder')
            self.feature_cols = checkpoint.get('feature_cols', [])
            self.seq_length = checkpoint.get('seq_length', 10)
            
            # 기본 파라미터 복원
            input_size = checkpoint.get('input_size', len(self.feature_cols))
            num_classes = len(self.encoder.classes_) if self.encoder else 0
            
            # [★ 핵심 수정] 모델 구조 및 하이퍼파라미터 자동 감지
            model_state = checkpoint['model_state']
            
            # 1. 임베딩 차원 및 직업 수 감지
            if 'job_embedding.weight' in model_state:
                emb_shape = model_state['job_embedding.weight'].shape
                # shape가 [N, D] 일 때, N=num_jobs+1, D=embedding_dim
                num_jobs = emb_shape[0] - 1
                embedding_dim = emb_shape[1]
                logger.info(f"파라미터 감지: num_jobs={num_jobs}, embedding_dim={embedding_dim}")
            else:
                num_jobs = 10
                embedding_dim = 8 # 기본값

            # 2. LSTM vs GRU 자동 감지 (가중치 크기 비율로 판단)
            # LSTM hidden size=256 -> weight size=1024 (4배)
            # GRU hidden size=256 -> weight size=768 (3배)
            is_gru = False
            for key in model_state.keys():
                if 'weight_ih_l0' in key: # 첫 번째 레이어 가중치 확인
                    weight_size = model_state[key].shape[0]
                    if weight_size == 768: # 256 * 3
                        is_gru = True
                    elif weight_size == 1024: # 256 * 4
                        is_gru = False
                    break
            
            # 모델 생성
            if is_gru:
                logger.info(f"모델 타입 감지: GRU ({file_path})")
                self.model = GRUModel(
                    input_size=input_size, 
                    hidden_size=256, 
                    num_layers=3, 
                    num_classes=num_classes, 
                    num_jobs=num_jobs, 
                    dropout=0.3,
                    embedding_dim=embedding_dim # 감지된 값 사용
                ).to(self.device)
            else:
                logger.info(f"모델 타입 감지: LSTM ({file_path})")
                self.model = LSTMModel(
                    input_size=input_size, 
                    hidden_size=256, 
                    num_layers=3, 
                    num_classes=num_classes, 
                    num_jobs=num_jobs, 
                    dropout=0.3,
                    embedding_dim=embedding_dim # 감지된 값 사용
                ).to(self.device)
            
            self.model.load_state_dict(model_state)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"AI 모델 로드 성공 (Device: {self.device})")
            return True
            
        except Exception as e:
            logger.error(f"AI 모델 로드 중 치명적 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False

    def predict(self, history_buffer, job_id):
        """
        시계열 데이터(history)를 기반으로 다음 행동 확률 예측
        
        Args:
            history_buffer (list): 정규화 전의 Feature Dictionary 리스트
            job_id (int): 현재 캐릭터의 직업 ID
            
        Returns:
            dict: {행동명: 확률} 또는 None
        """
        if not self.is_loaded or not self.model:
            return None
            
        try:
            # 1. 데이터 프레임 변환 및 결측 컬럼 채우기
            df = pd.DataFrame(list(history_buffer))
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # 2. 스케일링
            feats_scaled = self.scaler.transform(df[self.feature_cols])
            
            # 3. 텐서 변환 (Batch Size 1)
            # 입력 형태: (1, seq_length, input_size)
            inp = torch.FloatTensor(np.array([feats_scaled])).to(self.device)
            job_tensor = torch.LongTensor([job_id]).to(self.device)
            
            # 4. 추론
            with torch.no_grad():
                out = self.model(inp, job_tensor)
                # (batch, seq, hidden) -> 마지막 시퀀스만 사용
                if out.dim() == 3:
                    out = out[:, -1, :]
                
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                
            # 5. 결과 매핑 (확률 상위 3개 반환 등은 호출 측에서 처리)
            # 여기서는 가장 높은 확률의 인덱스와 확률 배열 자체를 반환하거나
            # 필요한 경우 LabelEncoder를 이용해 해석된 결과를 반환할 수도 있음.
            
            return probs # 확률 배열 반환
            
        except Exception as e:
            logger.error(f"모델 추론 오류: {e}")
            return None

    def decode_action(self, index):
        """인덱스를 실제 행동 이름으로 변환"""
        if self.encoder:
            return self.encoder.inverse_transform([index])[0]
        return "None"