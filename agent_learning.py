import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from simple_shaped_env import SimpleTaxiEnv

# ✅ 自訂特徵萃取器：LSTM + FC
class CustomLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, lstm_hidden_size=64):
        input_dim = observation_space.shape[0]  # obs shape: (14,)
        super().__init__(observation_space, features_dim=lstm_hidden_size)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, lstm_hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_seq = observations.unsqueeze(1)  # (B, 1, obs_dim)
        lstm_out, _ = self.lstm(obs_seq)
        return self.fc(lstm_out[:, -1, :])    # 最後時間步輸出

# ✅ 自訂 Policy
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomLSTMFeatureExtractor,
            features_extractor_kwargs=dict(lstm_hidden_size=64),
            **kwargs
        )

# ✅ 建立環境
env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

# ✅ 初始化 PPO 模型
model = PPO(
    policy=CustomLSTMPolicy,
    env=env,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    tensorboard_log="./logs"
)

# ✅ 開始訓練
model.learn(total_timesteps=100000)

# ✅ 儲存模型
model.save("ppo_taxi_model")
print("✅ 模型已儲存為 ppo_taxi_model")

# ✅ 評估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"📊 測試結果：Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
