import torch
import torch.nn as nn
import numpy as np
import random

# 跟 policy_net 一模一樣的結構
class PPOActor(nn.Module):
    def __init__(self, input_dim=16, action_dim=6):
        super(PPOActor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# 載入訓練好的 MLP policy_net
device = torch.device("cpu")
model = PPOActor(input_dim=16, action_dim=6)
model.load_state_dict(torch.load("ppo_policy_only.pth", map_location=device))
model.eval()

# 前處理函數
def preprocess_state(state):
    if isinstance(state, tuple):
        state = state[0]
    state = np.array(state, dtype=np.float32)
    return torch.tensor(state, dtype=torch.float32).to(device)

# 推論行動
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
