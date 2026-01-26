# Hunt_Train_v2/modules/physics_model.py

import torch
import torch.nn as nn

# [기존 클래스 유지]
class HybridPhysicsNet(nn.Module):
    def __init__(self, num_actions=16):
        super(HybridPhysicsNet, self).__init__()
        self.velocity = nn.Embedding(num_actions, 2)
        self.gravity = nn.Parameter(torch.tensor([5.0]))
        self.action_emb = nn.Embedding(num_actions, 16)
        self.residual_net = nn.Sequential(
            nn.Linear(16 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, action_idx, is_ground):
        base_vel = self.velocity(action_idx)
        act_vec = self.action_emb(action_idx)
        if is_ground.dim() == 1:
            is_ground = is_ground.unsqueeze(1)
        res_input = torch.cat([act_vec, is_ground], dim=1)
        residual = self.residual_net(res_input)
        return base_vel + residual, self.gravity

# [▼ 추가] 로그 기반으로 복원한 구버전 모델 클래스
class LegacyPhysicsNet(nn.Module):
    def __init__(self, num_actions=41):
        super(LegacyPhysicsNet, self).__init__()
        
        # [수정] 차원을 2 -> 3으로 변경 (로그 기반)
        # 파일에 저장된 가중치 크기 [41, 3]에 맞춤
        self.physics_params = nn.Embedding(num_actions, 3)
        
        self.action_emb = nn.Embedding(num_actions, 8)
        
        self.residual_net = nn.Sequential(
            nn.Linear(8 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        # gravity는 파일에 없으므로 기본값 5.0 사용 (register_buffer로 등록)
        self.register_buffer('gravity', torch.tensor([5.0]))

    def forward(self, action_idx, is_ground):
        # [수정] 3차원 파라미터에서 앞의 2개(vx, vy)만 잘라서 사용
        params = self.physics_params(action_idx)
        base_vel = params[:, :2] 
        
        act_vec = self.action_emb(action_idx)
        
        if is_ground.dim() == 1:
            is_ground = is_ground.unsqueeze(1)
            
        res_input = torch.cat([act_vec, is_ground], dim=1)
        residual = self.residual_net(res_input)
        
        return base_vel + residual, self.gravity