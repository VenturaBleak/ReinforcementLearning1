import random
import gym
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import trange
import os

from utils import PolynomialEpsilonDecay
from model import QNetwork

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = {
            'state': torch.zeros((capacity, state_dim), dtype=torch.float32),
            'action': torch.zeros(capacity, dtype=torch.int64),
            'reward': torch.zeros(capacity, dtype=torch.float32),
            'next_state': torch.zeros((capacity, state_dim), dtype=torch.float32),
            'done': torch.zeros(capacity, dtype=torch.bool)
        }
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer['state'][self.position] = state  # state is already a tensor
        self.buffer['action'][self.position] = torch.tensor(action, dtype=torch.int64)
        self.buffer['reward'][self.position] = torch.tensor(reward, dtype=torch.float32)
        self.buffer['next_state'][self.position] = next_state
        self.buffer['done'][self.position] = torch.tensor(done, dtype=torch.bool)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = torch.randint(0, self.capacity, (batch_size,))
        return (
            self.buffer['state'][indices],
            self.buffer['action'][indices],
            self.buffer['reward'][indices],
            self.buffer['next_state'][indices],
            self.buffer['done'][indices]
        )

    def clear(self):
        self.position = 0

    def __len__(self):
        if self.position < self.capacity:
            return self.position
        else:
            return self.capacity


def offline_train(environment, episodes, alpha, gamma, epsilon_decay_scheme, repeats, buffer_size=50000, batch_size=32):
    net = QNetwork(environment.observation_space.n, environment.action_space.n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    for cycle in trange(repeats, desc="Training Cycles", unit="cycle"):
        buffer = ReplayBuffer(buffer_size, env.observation_space.n)  # Modified this line
        outcomes = []
        epsilon = epsilon_decay_scheme.start_epsilon

        # Interaction Phase
        while buffer.__len__() + 1 < buffer_size:
            # print(buffer.__len__() + 1, buffer_size)
            state = environment.reset()
            state = state[0]
            done = False

            while not done:
                state_tensor = torch.eye(environment.observation_space.n)[state].unsqueeze(0).to(device)

                if np.random.random() < epsilon:
                    action = environment.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(net(state_tensor)).item()

                new_state, reward, done, _, info = environment.step(action)
                new_state_tensor = torch.eye(environment.observation_space.n)[new_state].unsqueeze(0).to(device)
                buffer.push(state_tensor, action, reward, new_state_tensor, done)

                # buffer.push(state, action, reward, new_state, done)
                state = new_state

                if reward:
                    outcomes.append("Success")
                else:
                    outcomes.append("Failure")

        # Learning Phase
        for _ in range((buffer_size // batch_size) + 1):  # Ensure total experiences used exceed 10000
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Compute Q-values and max Q-values
            q_values_next = net(next_states)
            max_q_values = torch.max(q_values_next, dim=-1)[0]

            target_rewards = rewards + gamma * max_q_values * (1 - dones.float())
            predicted_rewards = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            loss = criterion(predicted_rewards, target_rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save the buffer contents to a file
        # make dir if not exists
        if not os.path.exists("data"):
            os.makedirs("data")
        buffer_save_path = os.path.join("data", f"replay_buffer_{cycle}.pkl")
        with open(buffer_save_path, "wb") as f:
            pickle.dump(buffer.buffer, f)

        epsilon = epsilon_decay_scheme.step()

        print(epsilon)

    return net, outcomes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Offline training with Deep Q-learning using neural network and replay buffer.")
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel_offline.pth"),
                        help="Path to save the neural network model.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of times to repeat the training process.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total number of training episodes.")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--start_epsilon", type=float, default=1.0, help="Starting exploration rate.")
    parser.add_argument("--end_epsilon", type=float, default=1e-2, help="Minimum exploration rate.")
    parser.add_argument("--power", type=float, default=1.2, help="Power for the polynomial decay.")
    args = parser.parse_args()

    epsilon_decay_scheme = PolynomialEpsilonDecay(args.start_epsilon, args.end_epsilon, args.repeats, args.power)

    # Load the map from the file
    with open(os.path.join("data","frozen_lake_map.pkl"), "rb") as f:
        loaded_map = pickle.load(f)

    # Use the loaded map to create the environment
    env = gym.make('FrozenLake-v1', is_slippery=False, desc=loaded_map)
    model, outcomes = offline_train(env, args.episodes, args.alpha, args.gamma, epsilon_decay_scheme, args.repeats)
    torch.save(model.state_dict(), args.model_path)