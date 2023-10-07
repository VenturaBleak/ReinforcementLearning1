import os.path
import gym
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from collections import deque
import random

from utils import PolynomialEpsilonDecay
from model import QNetwork

class DQNAgent:
    def __init__(self, env, Qnet, episodes, alpha, gamma, epsilon_decay_scheme, model_path):
        self.env = env
        self.Qnet = Qnet
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

        self.memory = deque(maxlen=2000)
        self.batch_size = 4

        self.episodes = episodes
        self.epsilon_decay_scheme = epsilon_decay_scheme
        self.epsilon = self.epsilon_decay_scheme.start_epsilon

        self.model_path = model_path

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.Qnet(state_tensor)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.Qnet(next_state_tensor))

            prediction = self.Qnet(state_tensor)[0, action]
            loss = self.criterion(prediction.unsqueeze(0), torch.tensor([target], dtype=torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = self.epsilon_decay_scheme.step()

    def train(self):
        rewards = []
        t = trange(self.episodes, desc="Simulating", unit="episodes")
        for episode in t:
            state = self.env.reset()[0]
            episode_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _, info = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                self.replay()

            # save the model every 100 episodes
            if episode % 100 == 0:
                torch.save(Qnet.state_dict(), self.model_path)
                print(f"Saved model to {self.model_path}")

            rewards.append(episode_reward)
            if len(rewards) > 100:
                avg_reward_last_100 = sum(rewards[-100:]) / 100
                t.set_postfix(Avg_Reward=avg_reward_last_100)

        return self.Qnet, rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online training with Q-learning using neural network.")
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel_off.pth"), help="Path to save or load the neural network model.")
    parser.add_argument("--load_model", action="store_true", default=False, help="Specify to load a pretrained model for continued training.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total number of training episodes.")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor.")
    parser.add_argument("--start_epsilon", type=float, default=1, help="Starting exploration rate.")
    parser.add_argument("--end_epsilon", type=float, default=1e-3, help="Minimum exploration rate.")
    parser.add_argument("--power", type=float, default=1.5, help="Power for the polynomial decay.")
    args = parser.parse_args()

    epsilon_decay_scheme = PolynomialEpsilonDecay(args.start_epsilon, args.end_epsilon, args.episodes, args.power)

    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    Qnet = QNetwork(input_dim, output_dim)

    if args.load_model:
        Qnet.load_state_dict(torch.load(args.model_path))
        print(f"Loaded model from {args.model_path}")

    agent = DQNAgent(env, Qnet, args.episodes, args.alpha, args.gamma, epsilon_decay_scheme, args.model_path)
    model, outcomes = agent.train()