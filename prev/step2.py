import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


'''
This file iterates upon the step1 file in a few ways.

Firstly, the file implements the gym enviornment in order to get actual state information.
Secondly, we are able to retreive legitimate values associated with the env for state_dim, action_dim, max_action
Thridly, we link the actor and critic networks to the GPU if available
Lastly, we get the state from the enviornment, give the state to the actor, and then give the enviornment the newly found action to step forward with. 

Notes:
* this example only has one state (the starting state)
'''
# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        action = self.max_action * torch.tanh(self.layer3(x)) 
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

# Initialize the Continuous Mountain Car environment
env = gym.make('MountainCarContinuous-v0')

# Get state and action dimensions from the environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize actor and critic networks
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim).to(device)

# Example of interacting with the environment
state = env.reset()[0]
state = np.array([state], dtype=np.float32)  # Convert the list of arrays to a single NumPy array
state = torch.tensor(state).to(device)  # Convert to a tensor and move to the device

# Get action from the actor network
action = actor(state)
action = action.detach().cpu().numpy()[0]

# Step the environment with the chosen action
next_state, reward, done, _, _ = env.step(action)

# State is a 2D vector [position of the car, velocity of the car]
print("State:", state.cpu().numpy())
# Action chosen by the actor network given the initial state [force applied]
print("Action:", action)
# Next State after the chosen action above
print("Next State:", next_state)
# Reward recieved after taking the action
print("Reward:", reward)
# Done indicates the episode has ended
# For mountaincar this is after the car reaches to goal or a certain amount of steps
# False=episode has not ended, True=episode has ended
print("Done:", done)