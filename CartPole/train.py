import os.path
import gym
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from utils import PolynomialEpsilonDecay
from model import QNetwork

def online_train(environment, Qnet, episodes, alpha, gamma, epsilon_decay_scheme, model_path):
    optimizer = optim.Adam(Qnet.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    rewards = []
    best_reward = 0
    epsilon = epsilon_decay_scheme.start_epsilon

    # Create the progress bar
    t = trange(episodes, desc="Simulating", unit="episodes")

    for episode in t:
        state = environment.reset()[0]
        episode_reward = 0  # Track the total reward for each episode
        done = False

        while not done:
            # Convert state to tensor and add batch dimension
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if np.random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(Qnet(state_tensor)).item()

            new_state, reward, done, _, info = environment.step(action)
            episode_reward += reward

            target = reward
            if not done:
                # Convert new_state to tensor and add batch dimension
                target = reward + gamma * torch.max(Qnet(torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)))

            prediction = Qnet(state_tensor)[0, action]
            loss = criterion(prediction.unsqueeze(0), torch.tensor([target], dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state

        rewards.append(episode_reward)
        epsilon = epsilon_decay_scheme.step()

        # Display average reward of the last 100 episodes on the progress bar
        if len(rewards) > 100:
            avg_reward_last_100 = sum(rewards[-100:]) / 100
            t.set_postfix(Avg_Reward = avg_reward_last_100)

            if avg_reward_last_100 > best_reward and episode % 100 == 0:
                best_reward = avg_reward_last_100

                torch.save(Qnet.state_dict(), model_path)
                print(f"Saved model at episode {episode} with average reward of {avg_reward_last_100}")




    return Qnet, rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online training with Q-learning using neural network.")
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel.pth"), help="Path to save or load the neural network model.")
    parser.add_argument("--load_model", action="store_true", default=False, help="Specify to load a pretrained model for continued training.")
    parser.add_argument("--episodes", type=int, default=5000, help="Total number of training episodes.")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor.")
    parser.add_argument("--start_epsilon", type=float, default=1, help="Starting exploration rate.")
    parser.add_argument("--end_epsilon", type=float, default=1e-3, help="Minimum exploration rate.")
    parser.add_argument("--power", type=float, default=1.5, help="Power for the polynomial decay.")
    args = parser.parse_args()

    epsilon_decay_scheme = PolynomialEpsilonDecay(args.start_epsilon, args.end_epsilon, args.episodes, args.power)

    env = gym.make('CartPole-v1')

    # Define Qnet structure
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    Qnet = QNetwork(input_dim, output_dim)

    # Load the model if specified
    if args.load_model:
        Qnet.load_state_dict(torch.load(args.model_path))
        print(f"Loaded model from {args.model_path}")

    model, outcomes = online_train(env, Qnet, args.episodes, args.alpha, args.gamma, epsilon_decay_scheme, args.model_path)