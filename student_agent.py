import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 與 MLP policy 結構一致的推論網路
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim=16, action_dim=6):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        return logits

# 載入權重
device = torch.device("cpu")
model = PPOActorCritic(input_dim=16, action_dim=6)
model.load_state_dict(torch.load("ppo_policy_only.pth", map_location=device))
model.eval()

# 前處理函數：把 obs 轉成 Tensor
def preprocess_state(state):
    if isinstance(state, tuple):
        state = state[0]
    state = np.array(state, dtype=np.float32)
    return torch.tensor(state, dtype=torch.float32).to(device)

# 取得行動
def get_action(obs):
    try:
        state = preprocess_state(obs)
        with torch.no_grad():
            logits = model(state)
            action = torch.argmax(logits).item()
        return int(action)
    except Exception as e:
        print(f"[Fallback Error] {e}")
        return random.choice([0, 1, 2, 3, 4, 5])
