import pickle
import numpy as np
import random

# 載入訓練好的 Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

# 將 dict 轉成預設值為 0 的 defaultdict 結構
from collections import defaultdict
Q_table = defaultdict(lambda: np.zeros(6), Q_table)

# 根據 obs 轉換為簡化狀態（需與訓練時一致）
def simplify_state(obs):
    return (
        int(obs[0]), int(obs[1]),
        int(obs[10]), int(obs[11]),
        int(obs[12]), int(obs[13])
    )

# 根據 Q-table 選擇動作
def get_action(obs):
    try:
        state = simplify_state(obs)
        action_values = Q_table[state]
        action = int(np.argmax(action_values))
    except Exception as e:
        print(f"[Warning] Using random action due to error: {e}")
        action = random.randint(0, 5)
    return action
