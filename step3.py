import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

'''
This file provides a great number of changes from step 2 including
* Providing a replay memory class to handle the batches of steps used by the actor and critic for their calculations (action and qvalue respectively)
* Creates not only the required Actor and 2 critic objects, but actor_target and critic_target objects as well which are used as the inbetween for the 
actor to the critic and the critic to the actor
* introduces hyperparameter values but doesn't import them from a file. The parameters are provded within the file itself as of now.
* implementing the training loop which in our example (td3) uses the dual-critic network, batches of steps, and delayed policy updates to further train the model
'''
# TD3
# Some key characteristsics of TD3
# two Q-Learning functions
# "Delayed" policy updates
# addition of noise for better exploration

# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ReplayMemory class
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Actor NN
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
        action = self.max_action * torch.tanh(self.layer3(x))  # Ensure the action is in the correct range 
        return action

# Critic NN
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
env = gym.make('Pendulum-v1')

# Get state and action dimensions from the environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize Actor Networks (Actor and Actor Target)
# Main actor network selects actions given the current state
actor = Actor(state_dim, action_dim, max_action).to(device)
# Actor target network is a delayed copy of the main actor network
# This provides a stable target action for the updating critic network, thus reducing variance during training
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())

# Initialize Critic Networks (Critic and Critic Target)
# Main critic network 1 and 2 estimate the Q-values given the state action pair, Q(s,a)
critic_1 = Critic(state_dim, action_dim).to(device)

# Critic target networks are a delayed copy of the main critic networks
# Used to compute Q-values during critic network updates
# Using delayed targets help stabalize the learning process and reduce overestimation bias
critic_target_1 = Critic(state_dim, action_dim).to(device)
critic_target_1.load_state_dict(critic_1.state_dict())

critic_2 = Critic(state_dim, action_dim).to(device)
critic_target_2 = Critic(state_dim, action_dim).to(device)
critic_target_2.load_state_dict(critic_2.state_dict())

# Optimizers
# Adjusts the weights and biases to minimize error (loss) between predicted and actual
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer_1 = optim.Adam(critic_1.parameters(), lr=0.001)
critic_optimizer_2 = optim.Adam(critic_2.parameters(), lr=0.001)

# Hyperparameters
# Discount factor used in calculations for future rewards
# Value of 0.99 means that future rewards are discounted by 1% per timestep
gamma = 0.99 
# Controls the rate of soft updates for target networks, lower value = slower updates
# This is not the same as learning rate
tau = 0.005
# Improves robustness and encourages exploration
policy_noise = 0.2
# Range within which the policy noise can be clipped
# Ensures that actions remain in a reasonable range, prevents excessive noise values from distorting learning
noise_clip = 0.5
# Frequency of delayed policy updates
# TD3 use a delayed update strategy, Actor network is updated every 2 critic updates
policy_freq = 2
# Experience replay
# Holds past experiences
replay_buffer = ReplayMemory(maxlen=10000)

# Training loop
for episode in range(1000):
    # Reset Environment 
    state = env.reset()[0]
    episode_reward = 0
    # Maximum of 200 step in an episode
    for step in range(200):
        state_tensor = np.array([state], dtype=np.float32) # convert list to np array
        state_tensor = torch.tensor(state).to(device)  # convert np array to tensor

        # Select action
        action = actor(state_tensor).detach().cpu().numpy()[0] # detach removes optimization graph from the object, improving speed for the calculation
        action = np.clip(action + np.random.normal(0, max_action * 0.1, size=action_dim), -max_action, max_action) # force the action to be in range 
        
        # Interact with the environment
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        replay_buffer.append((state, action, reward, next_state, done)) # add step to replay_buffer
        
        # Sample from replay buffer if replay buffer proper size
        if len(replay_buffer) > 1000:
            minibatch = replay_buffer.sample(100)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Must convert lists of numpy arrays to numpy arrays otherise "too slow" warning
            # Convert lists of numpy arrays to numpy arrays
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            # Convert numpy arrays to tensors
            states = torch.tensor(states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).unsqueeze(1).to(device)
            next_states = torch.tensor(next_states).to(device)
            dones = torch.tensor(dones).unsqueeze(1).to(device)

            # Compute target actions with added noise
            noise = torch.clamp(torch.randn_like(actions) * policy_noise, -noise_clip, noise_clip)
            next_actions = torch.clamp(actor_target(next_states) + noise, -max_action, max_action)
            
            # Compute target Q-values
            target_Q1 = critic_target_1(next_states, next_actions)
            target_Q2 = critic_target_2(next_states, next_actions)
            target_Q = rewards + gamma * (1 - dones) * torch.min(target_Q1, target_Q2).detach() # calculate final q value using critic 1 and critic 2's minimum
            
            # Update critic networks
            current_Q1 = critic_1(states, actions)
            current_Q2 = critic_2(states, actions)
            critic_loss_1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
            critic_loss_2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
            
            critic_optimizer_1.zero_grad()
            critic_optimizer_2.zero_grad()
            critic_loss_1.backward() # computes the gradient for the critics
            critic_loss_2.backward()
            critic_optimizer_1.step()
            critic_optimizer_2.step()
            
            # Delayed policy updates
            if step % policy_freq == 0:
                actor_loss = -critic_1(states, actor(states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(critic_1.parameters(), critic_target_1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(critic_2.parameters(), critic_target_2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        if done:
            break
        state = next_state
    
    print(f"Episode {episode + 1}, Reward: {episode_reward}, States: {state}, Actions: {action}")
    # This is final state/action of each episode, to just see if values are in correct range
    #print(f"States: {state}")
    #print(f"Actions: {action}")

env.close()