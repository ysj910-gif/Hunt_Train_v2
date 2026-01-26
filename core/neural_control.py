# core/neural_control.py

import torch
import torch.nn as nn
import json
import joblib
import numpy as np
from collections import deque

# --- 학습 코드와 동일한 모델 클래스 정의 ---
class MovementPolicyPro(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MovementPolicyPro, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(
            input_size=hidden_size // 2, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        last_out = self.ln(out[:, -1, :])
        logits = self.fc(last_out)
        return self.sigmoid(logits)

class NeuralController:
    def __init__(self, model_path="movement_policy_pro.pth", meta_path="model_meta_pro.json", scaler_path="scaler_pro.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [수정 1] 변수 기본값 미리 설정 (안전장치)
        self.seq_len = 60 
        self.loaded = False
        self.feature_cols = []
        self.target_keys = []
        
        try:
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
            
            self.feature_cols = self.meta['feature_cols']
            self.target_keys = self.meta['target_keys']
            self.seq_len = self.meta['seq_length'] # 파일에서 읽은 값으로 업데이트
            
            self.scaler = joblib.load(scaler_path)
            
            # Pro 모델로 초기화
            self.model = MovementPolicyPro(
                input_size=len(self.feature_cols),
                hidden_size=self.meta['hidden_size'],
                num_layers=self.meta['num_layers'],
                num_classes=len(self.target_keys)
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.loaded = True
            print("[NeuralController] Model loaded successfully.")
            
        except Exception as e:
            # 로드 실패 시 로그만 출력하고 프로그램이 죽지 않게 함
            print(f"[NeuralController] Failed to load model (AI Disabled): {e}")
            self.loaded = False
            
        # [수정 2] self.seq_len이 위에서 초기화되었으므로 안전함
        self.history = deque(maxlen=self.seq_len)

    def predict(self, state_dict, threshold=0.4):
        if not self.loaded: return []

        features = []
        for col in self.feature_cols:
            features.append(state_dict.get(col, 0.0))
            
        features = np.array(features).reshape(1, -1)
        scaled_feat = self.scaler.transform(features)[0]
        
        self.history.append(scaled_feat)
        while len(self.history) < self.seq_len:
            self.history.append(scaled_feat)
            
        input_tensor = torch.FloatTensor([list(self.history)]).to(self.device)
        
        with torch.no_grad():
            probs = self.model(input_tensor)[0]
            
        active_keys = []
        for i, prob in enumerate(probs):
            if prob.item() > threshold:
                active_keys.append(self.target_keys[i])
                
        return active_keys