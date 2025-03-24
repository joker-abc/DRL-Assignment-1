import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from simple_shaped_env_v2 import MiniTaxiEnv as SimpleTaxiEnv

# 建立環境
env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

# 使用標準 MLP policy 訓練 PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1
)

# 開始訓練
model.learn(total_timesteps=100000)

# 評估訓練成果
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 🔥 儲存 MLP actor 的參數（對應 student_agent.py 的結構）
torch.save(model.policy.mlp_extractor.policy_net.state_dict(), "ppo_policy_only.pth")
