import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO inference.
    Takes 16-dim state and returns action logits and value.
    """
    def __init__(self, input_dim=16, action_dim=6):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
model = PPOActorCritic(input_dim=16, action_dim=6)
device = "cpu"
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

def preprocess_state(state):
    try:
        state_tuple = state[0]          
        state_list = list(state_tuple)
    except:
        state_list = list(state)
    if len(state_list) != 16:
        raise ValueError(f"Expected state to have 16 elements, got {len(state_list)}")
    return torch.tensor(state_list, dtype=torch.float32, device=device)

def get_action(obs):
    try:
        state = preprocess_state(obs)
        with torch.no_grad():
            logits, _ = model(state)
            action = torch.argmax(logits).item()
    except:
        action = random.choice([0, 1])
    return action

    
        
        

        
        
    
