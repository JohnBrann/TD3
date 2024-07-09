import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yaml
import argparse
from collections import deque


'''
This file differs from step3 in a few ways

* Imports the hyperparameters from the provided hyperparameters.yml file, cutting down on variable declarations. 
* Implements the agent class which utilizes the run() function to create the nn required for training and runs the training loop
* Begins the process of including command line arguments to change whether we are training/testing
'''

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

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.replay_buffer_size = hyperparameters['replay_buffer_size']
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.policy_noise = hyperparameters['policy_noise']
        self.noise_clip = hyperparameters['noise_clip']
        self.policy_freq = hyperparameters['policy_freq']

        self.replay_buffer = ReplayMemory(maxlen=self.replay_buffer_size)
    
    def run (self):
        # Initialize the Continuous Mountain Car environment
        env = gym.make(self.env_id)

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
        # Training loop
        for episode in range(1000):
            # Reset Environment 
            state = env.reset()[0]
            episode_reward = 0
            # Maximum of 200 step in an episode
            for step in range(200):
                state_tensor = np.array([state], dtype=np.float32)
                state_tensor = torch.tensor(state).to(device)
                
                # Select action
                action = actor(state_tensor).detach().cpu().numpy()[0]
                action = np.clip(action + np.random.normal(0, max_action * 0.1, size=action_dim), -max_action, max_action)
                
                # Interact with the environment
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                # Sample from replay buffer
                if len(self.replay_buffer) > 1000:
                    minibatch = self.replay_buffer.sample(100)
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
                    noise = torch.clamp(torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
                    next_actions = torch.clamp(actor_target(next_states) + noise, -max_action, max_action)
                    
                    # Compute target Q-values
                    target_Q1 = critic_target_1(next_states, next_actions)
                    target_Q2 = critic_target_2(next_states, next_actions)
                    target_Q = rewards + self.gamma * (1 - dones) * torch.min(target_Q1, target_Q2).detach()
                    
                    # Update critic networks
                    current_Q1 = critic_1(states, actions)
                    current_Q2 = critic_2(states, actions)
                    critic_loss_1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
                    critic_loss_2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
                    
                    critic_optimizer_1.zero_grad()
                    critic_optimizer_2.zero_grad()
                    critic_loss_1.backward()
                    critic_loss_2.backward()
                    critic_optimizer_1.step()
                    critic_optimizer_2.step()
                    
                    # Delayed policy updates
                    if step % self.policy_freq == 0:
                        actor_loss = -critic_1(states, actor(states)).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        
                        # Soft update target networks
                        for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        
                        for param, target_param in zip(critic_1.parameters(), critic_target_1.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        
                        for param, target_param in zip(critic_2.parameters(), critic_target_2.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                if done:
                    break
                state = next_state
            
            print(f"Episode {episode + 1}, Reward: {episode_reward}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Specify the hyperparameter set to use from the YAML file')
    args = parser.parse_args()
    
    # Initialize agent with specified hyperparameters
    agent = Agent(hyperparameter_set=args.hyperparameters)
    agent.run()

# env.close()