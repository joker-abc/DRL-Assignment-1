# import numpy as np
# import pickle
# import random
# from simple_shaped_env import SimpleShapeEnv as SimpleTaxiEnv
# from collections import defaultdict

# # =============== 超參數設定 ===============
# EPISODES = 3000      # 總回合數，可再視情況加大
# ALPHA = 0.1          # 學習率
# GAMMA = 0.95         # 折扣因子
# EPSILON = 1.0        # 初始探索率
# EPSILON_MIN = 0.1    # 最低探索率
# EPSILON_DECAY = 0.9995  # 探索率衰減

# # 透過 defaultdict(lambda: np.zeros(6)) 來存 Q-value
# # 其中 key 為 (row, col, obstN, obstS, obstE, obstW)
# # value 為對應 6 個動作 (0~5) 的 Q-value
# Q_table = defaultdict(lambda: np.zeros(6))

# def simplify_state(obs):
#     """
#     根據 obs: 
#     obs[0] = taxi_row
#     obs[1] = taxi_col
#     ...
#     obs[10] = obstacle_north
#     obs[11] = obstacle_south
#     obs[12] = obstacle_east
#     obs[13] = obstacle_west
#     ...
#     這裡只取 (row, col, obstN, obstS, obstE, obstW)
#     """
#     agent_row = int(obs[0])
#     agent_col = int(obs[1])
#     obst_north = int(obs[10])
#     obst_south = int(obs[11])
#     obst_east  = int(obs[12])
#     obst_west  = int(obs[13])

#     return (agent_row, agent_col, obst_north, obst_south, obst_east, obst_west)

# # =============== 建立環境並開始訓練 ===============
# env = SimpleTaxiEnv(fuel_limit=5000)

# for episode in range(EPISODES):
#     obs, _ = env.reset()
#     state = simplify_state(obs)
#     total_reward = 0
#     done = False

#     while not done:
#         # ============= e-greedy 策略選擇動作 =============
#         if random.random() < EPSILON:
#             action = random.randint(0, 5)
#         else:
#             action = np.argmax(Q_table[state])

#         # ============= 執行動作並得到回饋 =============
#         next_obs, reward, done, _ = env.step(action)
#         next_state = simplify_state(next_obs)

#         # ============= Q-Learning 更新 =============
#         old_value = Q_table[state][action]
#         next_max = np.max(Q_table[next_state])
#         new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
#         Q_table[state][action] = new_value

#         # 更新狀態和回合分數
#         state = next_state
#         total_reward += reward

#     # ============= 每回合結束後，更新 epsilon =============
#     EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

#     # 每 100 回合印一次成果
#     if (episode + 1) % 100 == 0:
#         print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")

# # ============= 將 Q-table 儲存下來 =============
# with open("q_table.pkl", "wb") as f:
#     pickle.dump(dict(Q_table), f)

# print("✅ Q-table saved to q_table.pkl")
import numpy as np
import pickle
import random
from simple_shaped_env import SimpleShapeEnv as SimpleTaxiEnv
from collections import defaultdict

EPISODES = 3000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995

Q_table = defaultdict(lambda: np.zeros(6))

def simplify_state(obs):
    return (
        int(obs[0]), int(obs[1]),
        int(obs[10]), int(obs[11]),
        int(obs[12]), int(obs[13])
    )

env = SimpleTaxiEnv(fuel_limit=5000)
episode_rewards = []

for episode in range(EPISODES):
    obs, _ = env.reset()
    state = simplify_state(obs)
    total_reward = 0
    done = False

    while not done:
        if random.random() < EPSILON:
            action = random.randint(0, 5)
        else:
            action = np.argmax(Q_table[state])

        next_obs, reward, done, _ = env.step(action)
        next_state = simplify_state(next_obs)

        old_value = Q_table[state][action]
        next_max = np.max(Q_table[next_state])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        Q_table[state][action] = new_value  # 可選：np.clip(new_value, -100, 100)

        state = next_state
        total_reward += reward

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_r = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}, Avg Reward: {avg_r:.2f}, Epsilon: {EPSILON:.3f}")

with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q_table), f)

print("✅ Q-table saved to q_table.pkl")
