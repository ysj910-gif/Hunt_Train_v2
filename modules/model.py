# Hunt_Train_v2/modules/model.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    # [수정] embedding_dim 파라미터 추가 (기본값 8)
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_jobs=10, dropout=0.3, embedding_dim=8):
        super(LSTMModel, self).__init__()
        
        # [수정] 고정값 8 대신 embedding_dim 사용
        self.job_embedding = nn.Embedding(num_jobs + 1, embedding_dim)
        
        # [수정] LSTM 입력 크기 계산 시 embedding_dim 사용
        self.lstm = nn.LSTM(
            input_size + embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, job_ids):
        job_emb = self.job_embedding(job_ids)
        job_emb = job_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, job_emb], dim=2)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# GRUModel도 동일하게 수정 (필요하다면)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_jobs=10, dropout=0.3, embedding_dim=8):
        super(GRUModel, self).__init__()
        self.job_embedding = nn.Embedding(num_jobs + 1, embedding_dim)
        self.gru = nn.GRU(
            input_size + embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, job_ids):
        job_emb = self.job_embedding(job_ids)
        job_emb = job_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, job_emb], dim=2)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out