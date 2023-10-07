import os.path

import gymnasium as gym
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import trange

from utils import PolynomialEpsilonDecay
from model import QNetwork

def online_train(environment, episodes, alpha, gamma, epsilon_decay_scheme):
    net = QNetwork(environment.observation_space.n, environment.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    outcomes = []
    epsilon = epsilon_decay_scheme.start_epsilon

    for _ in trange(episodes, desc="Simulating", unit = "episodes"):
        state = environment.reset()
        state = state[0]
        done = False
        outcomes.append("Failure")

        while not done:
            state_tensor = torch.eye(environment.observation_space.n)[state].unsqueeze(0)

            if np.random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(net(state_tensor)).item()

            new_state, reward, done, _, info = environment.step(action)
            # one-hot encoding of the new state, and unsqueeze to add a batch dimension
            # the result is a tensor of shape (1, env.observation_space.n)
            new_state_tensor = torch.eye(environment.observation_space.n)[new_state].unsqueeze(0)

            target = reward
            if not done:
                target = reward + gamma * torch.max(net(new_state_tensor))

            prediction = net(state_tensor)[0, action]
            loss = criterion(prediction.unsqueeze(0), torch.tensor([target], dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state

            if reward:
                outcomes[-1] = "Success"

        epsilon = epsilon_decay_scheme.step()

    return net, outcomes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online training with Q-learning using neural network.")
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel.pth"),
                        help="Path to save the neural network model.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total number of training episodes.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--start_epsilon", type=float, default=1.0, help="Starting exploration rate.")
    parser.add_argument("--end_epsilon", type=float, default=1e-5, help="Minimum exploration rate.")
    parser.add_argument("--power", type=float, default=1.5, help="Power for the polynomial decay.")
    args = parser.parse_args()

    epsilon_decay_scheme = PolynomialEpsilonDecay(args.start_epsilon, args.end_epsilon, args.episodes, args.power)

    # Load the map from the file
    with open(os.path.join("data","frozen_lake_map.pkl"), "rb") as f:
        loaded_map = pickle.load(f)

    # Use the loaded map to create the environment
    env = gym.make('FrozenLake-v1', is_slippery = False, desc=loaded_map)
    model, outcomes = online_train(env, args.episodes, args.alpha, args.gamma, epsilon_decay_scheme)

    torch.save(model.state_dict(), args.model_path)