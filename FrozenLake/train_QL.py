import gymnasium as gym
import numpy as np
import argparse
import pickle
from tqdm import trange
import os

from utils import PolynomialEpsilonDecay  # Assuming utils.py is in the same directory as train_QL.py

def online_train(environment, episodes, alpha, gamma, epsilon_decay_scheme):
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

    # print table dimensions
    print(qtable.shape)

    outcomes = []

    epsilon = epsilon_decay_scheme.start_epsilon

    for _ in trange(episodes, desc="Simulating", unit = "episodes"):
        state = environment.reset()
        state = state[0]
        done = False
        outcomes.append("Failure")

        while not done:
            if np.random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            new_state, reward, done, _, info = environment.step(action)

            # Update Q(s,a)
            qtable[state, action] = qtable[state, action] + \
                                    alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            state = new_state

            if reward:
                outcomes[-1] = "Success"

        epsilon = epsilon_decay_scheme.step()

    print()
    print('===========================================')
    print('Q-table after training:')
    print(qtable)
    print('===========================================')

    return qtable, outcomes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online training with Q-learning.")
    parser.add_argument("--qtable_path", type=str, default=os.path.join("data","qtable.pkl"), help="Path to save the Q-table.")
    parser.add_argument("--episodes", type=int, default=20000, help="Total number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--start_epsilon", type=float, default=1.0, help="Starting exploration rate.")
    parser.add_argument("--end_epsilon", type=float, default=1e-3, help="Minimum exploration rate.")
    parser.add_argument("--power", type=float, default=1.1, help="Power for the polynomial decay.")
    args = parser.parse_args()

    epsilon_decay_scheme = PolynomialEpsilonDecay(args.start_epsilon, args.end_epsilon, args.episodes, args.power)

    # Load the map from the file
    with open(os.path.join("data","frozen_lake_map.pkl"), "rb") as f:
        loaded_map = pickle.load(f)

    # Use the loaded map to create the environment
    env = gym.make('FrozenLake-v1', is_slippery = False, desc=loaded_map)

    qtable, outcomes = online_train(env, args.episodes, args.alpha, args.gamma, epsilon_decay_scheme)

    with open(args.qtable_path, 'wb') as f:
        pickle.dump(qtable, f)