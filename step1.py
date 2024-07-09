import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        action = self.max_action * torch.tanh(self.layer3(x))  # Ensure action is in the correct range
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.layer1(torch.cat([state, action], 1)))
        x = torch.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value

# Example usage:
state_dim = 3
action_dim = 1
max_action = 1.0

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)

# Sample state and action
state = torch.tensor([[0.1, 0.2, 0.3]])
action = actor(state)
q_value = critic(state, action)

print("Action:", action)
print("Q-value:", q_value)