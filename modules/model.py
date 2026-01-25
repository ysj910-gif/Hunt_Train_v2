#modules\model.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_jobs=10, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # 직업(Job) 임베딩 레이어
        # 직업 ID(0~num_jobs)를 8차원 벡터로 변환하여 특성에 추가
        self.job_embedding = nn.Embedding(num_jobs + 1, 8) 
        
        # LSTM 레이어
        # 입력 차원 = 센서 데이터(input_size) + 직업 정보(8)
        self.lstm = nn.LSTM(
            input_size + 8, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # 분류기 (Fully Connected Layer)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, job_ids):
        # x: (batch, seq_length, input_size)
        # job_ids: (batch,)
        
        # 1. 직업 ID를 임베딩 벡터로 변환
        job_emb = self.job_embedding(job_ids) # (batch, 8)
        
        # 2. 시퀀스 길이만큼 직업 정보 복제 (매 프레임마다 직업 정보 추가)
        # (batch, 1, 8) -> (batch, seq_length, 8)
        job_emb = job_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 3. 입력 데이터와 직업 정보 결합
        x = torch.cat([x, job_emb], dim=2) # (batch, seq, input+8)
        
        # 4. LSTM 통과
        out, _ = self.lstm(x)
        
        # 5. 마지막 시퀀스의 결과로 예측
        # out: (batch, seq, hidden)
        out = self.fc(out) # (batch, seq, num_classes)
        
        return out