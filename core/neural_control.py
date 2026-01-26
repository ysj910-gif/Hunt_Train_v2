# core/neural_control.py

import torch
import torch.nn as nn
import json
import joblib
import numpy as np
from collections import deque
from utils.logger import logger, trace_logic

class MovementPolicyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MovementPolicyGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

class NeuralController:
    def __init__(self, model_path="movement_policy.pth", meta_path="model_meta.json", scaler_path="scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 메타데이터 로드
        try:
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
            self.feature_cols = self.meta['feature_cols']
            self.target_keys = self.meta['target_keys']
            self.seq_len = self.meta['seq_length']
            
            # Scaler 로드
            self.scaler = joblib.load(scaler_path)
            
            # 모델 로드
            self.model = MovementPolicyGRU(
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
            print(f"[NeuralController] Failed to load model: {e}")
            self.loaded = False
            
        # 히스토리 버퍼 (Sliding Window용)
        self.history = deque(maxlen=self.seq_len)

    @trace_logic
    def predict(self, state_dict, threshold=0.5):
        """
        현재 상태(state_dict)를 받아 누를 키 목록을 반환
        """
        if not self.loaded:
            return []

        # 1. Feature 추출
        features = []
        for col in self.feature_cols:
            val = state_dict.get(col, 0.0)
            if val is None: val = 0.0
            features.append(val)
            
        # 2. 스케일링
        features = np.array(features).reshape(1, -1)
        scaled_feat = self.scaler.transform(features)[0]
        
        # 3. 히스토리 업데이트
        self.history.append(scaled_feat)
        
        # 데이터가 충분히 쌓이지 않았으면 마지막 상태 복제
        while len(self.history) < self.seq_len:
            self.history.append(scaled_feat)
            
        # 4. 모델 추론
        input_tensor = torch.FloatTensor([list(self.history)]).to(self.device)
        
        with torch.no_grad():
            probs = self.model(input_tensor)[0] # (num_classes, )
            
        # 5. 결과 해석 (Multi-label)
        active_keys = []
        for i, prob in enumerate(probs):
            if prob.item() > threshold:
                key_name = self.target_keys[i]
                active_keys.append(key_name)
                
        return active_keys