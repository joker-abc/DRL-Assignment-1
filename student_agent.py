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

# import torch
# import torch.nn as nn
# import numpy as np

# class DQN(nn.Module):
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(4, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 6)
#         )

#     def forward(self, x):
#         return self.net(x)

# # 載入模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DQN().to(device)
# model.load_state_dict(torch.load("dqn_model.pt", map_location=device))
# model.eval()

# def get_action(obs):
#     obstacle_obs = torch.tensor(obs[10:14], dtype=torch.float32).to(device)
#     with torch.no_grad():
#         q_values = model(obstacle_obs)
#     return torch.argmax(q_values).item()


import numpy as np
import pickle
import random

# 載入訓練好的 Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

# 使用與訓練相同的狀態定義：row, col, obst_N, obst_S, obst_E, obst_W
def simplify_state(obs):
    agent_row = obs[0]
    agent_col = obs[1]
    obst_north = obs[10]
    obst_south = obs[11]
    obst_east  = obs[12]
    obst_west  = obs[13]
    return (agent_row, agent_col, obst_north, obst_south, obst_east, obst_west)

def get_action(obs):
    state = simplify_state(obs)

    if state in Q_table:
        return int(np.argmax(Q_table[state]))
    else:
        # fallback：若遇未知狀態，嘗試避開障礙的方向（從 obs[10:14] 得知）
        safe_actions = []
        directions = [0, 1, 2, 3]  # Down, Up, Right, Left
        for i, is_blocked in enumerate(obs[10:14]):
            if is_blocked == 0:
                safe_actions.append(directions[i])
        if safe_actions:
            return random.choice(safe_actions)
        return random.choice(directions)  # 全部擋住，隨便走
