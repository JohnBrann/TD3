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

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')


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

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# Actor NN
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=300):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

        self.apply(init_weights)  # Apply the weight initialization

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        action = self.max_action * torch.tanh(self.layer3(x))  # Ensure the action is in the correct range 
        return action

# # Critic NN
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=400):
#         super(Critic, self).__init__()
#         self.layer1 = nn.Linear(action_dim + state_dim, int(hidden_dim / 2))
#         self.layer2 = nn.Linear(action_dim, int(hidden_dim / 2))
#         self.layer3 = nn.Linear(hidden_dim, hidden_dim)
#         self.layer4 = nn.Linear(hidden_dim, 1)

#         self.apply(init_weights)  # Apply the weight initialization

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = torch.relu(self.layer1(torch.cat([state, action], 1)))
#         x = torch.relu(self.layer2(x))
#         q_value = self.layer3(x)
#         return q_value

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=200):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)  # Output a single Q-value

        self.apply(init_weights)  # Apply the weight initialization

    def forward(self, state, action):
        # Concatenate state and action tensors along the batch dimension
        x = torch.cat([state, action], dim=1)  # dim=1 means concatenate along the columns
        x = torch.relu(self.layer1(x))  # Pass through first layer with ReLU activation
        x = torch.relu(self.layer2(x))  # Pass through second layer with ReLU activation
        q_value = self.layer3(x)  # Get the Q-value from the last layer
        return q_value
    
    
class Agent():
    def __init__(self, hyperparameter_set, is_training):
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
        self.learning_rate = hyperparameters['learning_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.episode_start_reward = hyperparameters['episode_start_reward']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        self.highest_reward = -np.inf
        self.is_training = is_training

        self.replay_buffer = ReplayMemory(maxlen=self.replay_buffer_size)

        # Initialize the environment
        self.env = gym.make(self.env_id, render_mode='human' if not is_training else None, **self.env_make_params)

        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.best_reward = -9999999
        self.max_steps = self.env.spec.max_episode_steps  # Dynamically get the truncation value

        # Initialize Actor Network
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)

        if is_training:
            # Initialize Actor Target Networks
            self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            # Initialize Critic Networks (Critic and Critic Target)
            self.critic_1 = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target_1 = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target_1.load_state_dict(self.critic_1.state_dict())
            self.critic_2 = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target_2 = Critic(self.state_dim, self.action_dim).to(device)
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())

    def run(self, is_training):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

            self.actor_optimizer = optim.Adam(self.actor.parameters(), self.learning_rate)
            self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), self.learning_rate)
            self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), self.learning_rate)
        else:
            self.actor.load_state_dict(torch.load(self.MODEL_FILE))
            self.actor.eval()

        for episode in itertools.count():
            state = self.env.reset()[0]
            episode_reward = self.episode_start_reward
            terminated = False
            truncated = False
            step_count = 0

            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # If the reward is not a result that we want we continue training
            while not terminated and not truncated:
                # Select action
                with torch.no_grad():
                    action = self.actor(state_tensor).detach().cpu().numpy()[0]
                action = np.clip(action + np.random.normal(0, self.max_action * 0.1, size=self.action_dim), -self.max_action, self.max_action)

                # Interact with the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if is_training:
                    self.replay_buffer.append((state, action, reward, next_state, terminated, truncated))
                    step_count += 1
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Update the tensor for the new state
                
            self.rewards_per_episode.append(episode_reward)

            if is_training:
                # save the model everytime we get a new highest reward
                if episode_reward > self.best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.2f} ({(episode_reward-self.best_reward)/self.best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(self.actor.state_dict(), self.MODEL_FILE)
                    self.best_reward = episode_reward
                
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(self.rewards_per_episode)
                    last_graph_update_time = current_time
                    
                if len(self.replay_buffer) > self.mini_batch_size:
                    self.optimize(step_count)
            
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    def optimize(self, step_count):
        # Sample from replay buffer and train
        # Check if is training and replay buffer contains more then n samples
        if self.is_training and len(self.replay_buffer) > 1000:
            # Takes minibatch of 100 episodes from the experience replay
            minibatch = self.replay_buffer.sample(self.mini_batch_size)
            states, actions, rewards, next_states, dones, truncs = zip(*minibatch)
            
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            truncs = np.array(truncs, dtype=np.float32)
            
            states = torch.tensor(states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).unsqueeze(1).to(device)
            next_states = torch.tensor(next_states).to(device)
            dones = torch.tensor(dones).unsqueeze(1).to(device)
            truncs = torch.tensor(truncs).unsqueeze(1).to(device)
            
            # Adding noise, similar to in DQN of selecting random action 
            noise = torch.clamp(torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(self.actor_target(next_states) + noise, -self.max_action, self.max_action)
            
            # Calculates target Q-values, 2 neural networks leading to less drastic changes in learning (takes min(q1, q2))
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = rewards + self.gamma * (1 - dones) * torch.min(target_Q1, target_Q2).detach()
            
            current_Q1 = self.critic_1(states, actions)
            current_Q2 = self.critic_2(states, actions)
            critic_loss_1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
            critic_loss_2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
            
            # Optimize critic networks
            self.critic_optimizer_1.zero_grad()
            self.critic_optimizer_2.zero_grad()
            critic_loss_1.backward()
            critic_loss_2.backward()
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.step()
            
            # Actor NN is updated based on the policy_freq the actor network is updated
            if step_count % self.policy_freq == 0:
                #  The loss for the actor network is the negative mean Q-value, the MSE
                #  The difference between predicted and the actual 
                actor_loss = -self.critic_1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # the target networks are then softly updated
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) *  target_param.data)

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
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Specify the hyperparameter set to use from the YAML file')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()
    
    # Initialize agent with specified hyperparameters
    td3 = Agent(hyperparameter_set=args.hyperparameters, is_training=args.train)
    td3.run(is_training=args.train)