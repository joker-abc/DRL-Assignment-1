import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from simple_shaped_env_v2 import MiniTaxiEnv as SimpleTaxiEnv 

# 1️⃣ Fixed Custom LSTM Feature Extractor
class LSTMFeatureEncoder(BaseFeaturesExtractor):
    """
    A custom feature extractor using LSTM for temporal learning in PPO.
    """
    def __init__(self, observation_space, lstm_hidden_size=64):
        input_dim = observation_space.shape[0]  # Fix: Correct input size
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, lstm_hidden_size)  

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Fix: Ensure observations are 3D (batch, seq_len=1, features)
        obs_seq = observations.unsqueeze(1)  

        # Pass through LSTM
        lstm_out, _ = self.lstm(obs_seq)
        lstm_out = lstm_out[:, -1, :]  # Take the last output from LSTM

        return self.fc(lstm_out)  

# 2️⃣ Define Custom LSTM Policy
class LSTMPolicy(ActorCriticPolicy):
    """
    Custom PPO policy that uses the LSTMFeatureEncoder.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(LSTMPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=LSTMFeatureEncoder,
            features_extractor_kwargs={'lstm_hidden_size': 64},
            **kwargs,
        )

# 3️⃣ Environment Setup
env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)

# 4️⃣ Train PPO with Custom LSTM
model = PPO(
    policy=LSTMPolicy,
    env=env,
    learning_rate=0.0003,
    n_steps=512,  
    batch_size=128,  
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1
)

model.learn(total_timesteps=100000)

model.save("ppo_taxi_model")

# 5️⃣ Evaluate the Model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
