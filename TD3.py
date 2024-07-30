import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import yaml
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import deque
from datetime import datetime, timedelta

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        transitions = random.sample(self.memory, sample_size)
        # Unzip the transitions into separate lists
        return zip(*transitions)  # This will return a tuple of lists: (state, action, next_state, reward, terminated)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

class Actor(Network):
    def __init__(self, state_dim, action_dim, max_action, actor_hidden_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, actor_hidden_dim)
        self.l2 = nn.Linear(actor_hidden_dim, actor_hidden_dim)
        self.l3 = nn.Linear(actor_hidden_dim, action_dim)
        self.max_action = max_action
        self.apply(self.init_weights)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(Network):
    def __init__(self, state_size, action_size, critic_hidden_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, critic_hidden_dim)
        self.l2 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.l3 = nn.Linear(critic_hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, critic_hidden_dim)
        self.l5 = nn.Linear(critic_hidden_dim, critic_hidden_dim)
        self.l6 = nn.Linear(critic_hidden_dim, 1)
        self.apply(self.init_weights)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1 forward pass
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2 forward pass
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class TD3Agent:
    def __init__(self, hyperparameter_set, is_training):
        # Load hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
        self.hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        # Set hyperparameters
        self.env_id = self.hyperparameters['env_id']
        self.replay_buffer_size = self.hyperparameters['replay_buffer_size']
        self.gamma = self.hyperparameters['gamma']
        self.tau = self.hyperparameters['tau']
        self.policy_noise = self.hyperparameters['policy_noise']
        self.noise_clip = self.hyperparameters['noise_clip']
        self.policy_freq = self.hyperparameters['policy_freq']
        self.learning_rate = self.hyperparameters['learning_rate']
        self.stop_on_reward = self.hyperparameters['stop_on_reward']
        self.mini_batch_size = self.hyperparameters['mini_batch_size']
        self.max_episodes = self.hyperparameters['max_episodes']
        self.actor_hidden_dim = self.hyperparameters['actor_hidden_dim']
        self.critic_hidden_dim = self.hyperparameters['critic_hidden_dim']
        self.env_make_params = self.hyperparameters.get('env_make_params', {})

        # Set file paths
        RUNS_DIR = 'runs'
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')

        # Added variables
        self.is_training =  is_training
        self.highest_reward = -np.inf
        self.total_iterations = 0
        self.replay_buffer = ReplayMemory(maxlen=self.replay_buffer_size)

        # Initialize the environment
        self.env = gym.make(self.env_id, render_mode='human' if not is_training else None, **self.env_make_params)
        
        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # More initializing
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.best_reward = -float('inf')
        self.max_steps = self.env.spec.max_episode_steps  # Dynamically get the truncation value

        # Initialize Actor Network
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, self.actor_hidden_dim).to(device)

        if is_training:
            # Initialize Actor Target Network
            self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action, self.actor_hidden_dim).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            # Initialize Critic Networks (Critic and Critic Target)
            self.critic = Critic(self.state_dim, self.action_dim, self.critic_hidden_dim).to(device)
            self.critic_target = Critic(self.state_dim, self.action_dim, self.critic_hidden_dim).to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Initialize optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

            # Clear log file at the start of each run
            open(self.LOG_FILE, 'w').close()
        else:
            self.actor.load_state_dict(torch.load(self.MODEL_FILE))
            self.actor.eval() 

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def run(self):
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            step_count = 0
            terminated = False

            while not terminated and step_count < self.max_steps:
                action = self.select_action(state)
                
                if self.is_training:
                    action = action + np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
                    action = np.clip(action, -self.max_action, self.max_action)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminated = terminated or truncated

                if self.is_training:
                    self.replay_buffer.append((state, action, next_state, reward, float(terminated)))
                    
                    # Train the agent every few steps, e.g., every 4 steps
                    if len(self.replay_buffer) >= self.mini_batch_size and step_count % 4 == 0:
                        self.train()

                state = next_state
                episode_reward += reward
                step_count += 1  # Increment step counter

            self.rewards_per_episode.append(episode_reward)
            if self.is_training:
                # Save the model every time we get a new highest reward
                if episode_reward > self.best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.2f} at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(self.actor.state_dict(), self.MODEL_FILE)
                    self.best_reward = episode_reward
                    self.save_graph(self.rewards_per_episode)

                if self.best_reward >= self.stop_on_reward:
                    print(f"Solved in {episode} episodes!")
                    break

        self.save_graph(self.rewards_per_episode)
        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Training finished. best reward:{episode_reward:0.2f}..."
        print(log_message)
        self.env.close()

    def train(self):
        if len(self.replay_buffer) < self.mini_batch_size:
            return

        # Sample a batch from memory
        state, action, next_state, reward, terminated = self.replay_buffer.sample(self.mini_batch_size)
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device)
        terminated = torch.FloatTensor(np.array(terminated)).to(device)

        # Select action according to policy and add clipped noise
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward.unsqueeze(1) + (1 - terminated.unsqueeze(1)) * self.gamma * target_Q.detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_iterations % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state))[0].mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Increment the total_iterations counter
        self.total_iterations += 1



    def save_graph(self, rewards_per_episode):
        fig, ax = plt.subplots()
        # Calculate mean rewards
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        # Plot rewards per episode
        ax.plot(rewards_per_episode, label='Rewards per Episode', color='tab:blue')
        # Plot mean rewards
        ax.plot(mean_rewards, label='Mean Rewards (100 episodes)', color='tab:orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rewards')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test TD3 model.')
    parser.add_argument('hyperparameters', help='Specify the hyperparameter set to use from the YAML file')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    # Initialize agent with specified hyperparameters
    td3_agent = TD3Agent(hyperparameter_set=args.hyperparameters, is_training=args.train)

    # Run the agent
    td3_agent.run()