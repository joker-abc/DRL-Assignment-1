# import numpy as np
# import pickle
# import random
# from simple_shaped_env import SimpleShapeEnv as SimpleTaxiEnv
# from collections import defaultdict

# EPISODES = 3000
# ALPHA = 0.1
# GAMMA = 0.95
# EPSILON = 1.0
# EPSILON_MIN = 0.1
# EPSILON_DECAY = 0.9995

# Q_table = defaultdict(lambda: np.zeros(6))

# def simplify_state(obs):
#     return (
#         int(obs[0]), int(obs[1]),
#         int(obs[10]), int(obs[11]),
#         int(obs[12]), int(obs[13])
#     )

# env = SimpleTaxiEnv(fuel_limit=5000)
# episode_rewards = []

# for episode in range(EPISODES):
#     obs, _ = env.reset()
#     state = simplify_state(obs)
#     total_reward = 0
#     done = False

#     while not done:
#         if random.random() < EPSILON:
#             action = random.randint(0, 5)
#         else:
#             action = np.argmax(Q_table[state])

#         next_obs, reward, done, _ = env.step(action)
#         next_state = simplify_state(next_obs)

#         old_value = Q_table[state][action]
#         next_max = np.max(Q_table[next_state])
#         new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
#         Q_table[state][action] = new_value  # 可選：np.clip(new_value, -100, 100)

#         state = next_state
#         total_reward += reward

#     EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
#     episode_rewards.append(total_reward)

#     if (episode + 1) % 100 == 0:
#         avg_r = np.mean(episode_rewards[-100:])
#         print(f"Episode {episode+1}, Avg Reward: {avg_r:.2f}, Epsilon: {EPSILON:.3f}")

# with open("q_table.pkl", "wb") as f:
#     pickle.dump(dict(Q_table), f)

# print("✅ Q-table saved to q_table.pkl")

import numpy as np
import pickle
import random
from simple_shaped_env import SimpleShapeEnv as SimpleTaxiEnv
from collections import defaultdict

# 超參數
EPISODES = 3000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995

# Q-table: 用障礙物方向組合作為 key
Q_table = defaultdict(lambda: np.zeros(6))

# 只取 obstacle 四個方向
def simplify_state(obs):
    obst_north = int(obs[10])
    obst_south = int(obs[11])
    obst_east  = int(obs[12])
    obst_west  = int(obs[13])
    return (obst_north, obst_south, obst_east, obst_west)

# 初始化環境
env = SimpleTaxiEnv(fuel_limit=5000)
episode_rewards = []

# 開始訓練
for episode in range(EPISODES):
    obs, _ = env.reset()
    state = simplify_state(obs)
    total_reward = 0
    done = False

    while not done:
        # ε-greedy 選擇行動
        if random.random() < EPSILON:
            action = random.randint(0, 5)
        else:
            action = np.argmax(Q_table[state])

        # 執行動作
        next_obs, reward, done, _ = env.step(action)
        next_state = simplify_state(next_obs)

        # Q-learning 更新
        old_value = Q_table[state][action]
        next_max = np.max(Q_table[next_state])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        Q_table[state][action] = new_value

        # 更新狀態
        state = next_state
        total_reward += reward

    # 探索率遞減
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_r = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}, Avg Reward: {avg_r:.2f}, Epsilon: {EPSILON:.3f}")

# 儲存訓練結果
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q_table), f)

print("✅ Q-table saved to q_table.pkl")
