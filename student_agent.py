# # Remember to adjust your student ID in meta.xml
# import numpy as np
# import pickle
# import random
# import gym

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.



import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.net(x)

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
model.load_state_dict(torch.load("dqn_model.pt", map_location=device))
model.eval()

def get_action(obs):
    obstacle_obs = torch.tensor(obs[10:14], dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = model(obstacle_obs)
    return torch.argmax(q_values).item()
