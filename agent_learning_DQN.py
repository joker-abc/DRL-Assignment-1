import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from simple_shaped_env import SimpleShapedEnv as SimpleTaxiEnv


# 超參數
EPISODES = 1000
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型輸入：4 維（上下左右障礙）
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 六個行動
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPSILON_START

for episode in range(EPISODES):
    obs, _ = env.reset()
    state = torch.tensor(obs[10:14], dtype=torch.float32).to(device)
    total_reward = 0
    done = False
    print("episode", episode)
    
    while not done:
        if random.random() < epsilon:
            action = random.randint(0, 5)
        else:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()
            

        next_obs, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_obs[10:14], dtype=torch.float32).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        done_tensor = torch.tensor([done], dtype=torch.float32).to(device)

        memory.push((state, action, reward_tensor, next_state, done_tensor))
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)

            s_batch = torch.stack(s_batch)
            a_batch = torch.tensor(a_batch, dtype=torch.long).to(device)
            r_batch = torch.stack(r_batch).squeeze()
            ns_batch = torch.stack(ns_batch)
            d_batch = torch.stack(d_batch).squeeze()

            q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
            next_q = target_net(ns_batch).max(1)[0].detach()
            target_q = r_batch + GAMMA * next_q * (1 - d_batch)

            loss = nn.MSELoss()(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

torch.save(policy_net.state_dict(), "dqn_model.pt")
print("✅ Model saved as dqn_model.pt")
