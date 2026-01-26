# train_movement_select.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# --- [High-Performance Config for RTX 4070 Super] ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# 하이퍼파라미터
SEQ_LENGTH = 60      # 과거 2초(60프레임)간의 동작 흐름
HIDDEN_SIZE = 1024   # 모델 표현력
NUM_LAYERS = 4       # 깊은 구조
BATCH_SIZE = 512     # 배치 크기
EPOCHS = 200         # 학습 횟수
LEARNING_RATE = 0.0001
DROPOUT = 0.3

# 피처 및 타겟 설정
FEATURE_COLS = [
    'player_x', 'player_y', 
    'delta_x', 'delta_y', 
    'dist_left', 'dist_right',
    'inv_dist_up', 'inv_dist_down', 'inv_dist_left', 'inv_dist_right',
    'ult_ready', 'sub_ready', 'kill_count'
]
TARGET_KEYS = ['left', 'right', 'up', 'down', 'jump', 'attack', 'ultimate', 'fountain', 'rope']
KEY_ALIAS = {'double_jump': 'jump', 'teleport': 'jump'}

class GameplayDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Pro 모델 구조
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
            dropout=DROPOUT
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

def parse_key_string(key_str):
    if pd.isna(key_str) or key_str == 'None':
        return np.zeros(len(TARGET_KEYS), dtype=int)
    active_keys = set()
    for p in str(key_str).split('+'):
        p = p.strip()
        p = KEY_ALIAS.get(p, p)
        if p in TARGET_KEYS:
            active_keys.add(p)
    return np.array([1 if k in active_keys else 0 for k in TARGET_KEYS], dtype=int)

def select_files():
    """파일 선택 대화상자를 띄웁니다."""
    root = tk.Tk()
    root.withdraw() # 메인 윈도우 숨김
    root.attributes('-topmost', True) # 창을 맨 위로
    
    print("학습할 CSV 파일들을 선택해주세요...")
    file_paths = filedialog.askopenfilenames(
        title="학습 데이터 선택 (CSV)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_paths

def main():
    # 1. 데이터 로드 (파일 선택)
    csv_files = select_files()
    
    if not csv_files:
        print("❌ 파일이 선택되지 않았습니다. 종료합니다.")
        return

    print(f"📂 Selected {len(csv_files)} files:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 필요한 컬럼이 있는지 확인
            if 'key_pressed' in df.columns:
                df_list.append(df)
            else:
                print(f"⚠️ Skipping {os.path.basename(f)}: 'key_pressed' column missing.")
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            
    if not df_list:
        print("학습 가능한 데이터가 없습니다.")
        return

    full_df = pd.concat(df_list, ignore_index=True).fillna(0)
    print(f"✅ Total Frames Loaded: {len(full_df)}")

    # 2. 전처리
    print("Preprocessing...")
    scaler = StandardScaler()
    for col in FEATURE_COLS:
        if col not in full_df.columns: full_df[col] = 0
    
    scaled_features = scaler.fit_transform(full_df[FEATURE_COLS].values)
    label_vectors = np.array([parse_key_string(k) for k in full_df['key_pressed'].values])

    # 3. 시퀀스 생성
    X_data, y_data = [], []
    print(f"Creating sequences (Length: {SEQ_LENGTH})...")
    
    # 데이터가 너무 많으면 메모리 부족 가능성 -> stride로 조절 가능
    stride = 1 
    for i in range(0, len(scaled_features) - SEQ_LENGTH, stride):
        X_data.append(scaled_features[i : i+SEQ_LENGTH])
        y_data.append(label_vectors[i+SEQ_LENGTH])
        
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    # 4. 학습 준비
    dataset = GameplayDataset(X_data, y_data)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 모델 생성
    model = MovementPolicyPro(
        input_size=len(FEATURE_COLS), 
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(TARGET_KEYS)
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Start Training on {device}...")

    # 5. 학습 루프
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for seqs, labels in loop:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f" > Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "movement_policy_pro.pth")
            
    # 메타데이터 저장
    meta_data = {
        "feature_cols": FEATURE_COLS,
        "target_keys": TARGET_KEYS,
        "seq_length": SEQ_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "model_type": "Pro"
    }
    with open("model_meta_pro.json", "w") as f:
        json.dump(meta_data, f)
    joblib.dump(scaler, "scaler_pro.pkl")
    print("\n✅ All saved successfully. (movement_policy_pro.pth)")

if __name__ == "__main__":
    main()