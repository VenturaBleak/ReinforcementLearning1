import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from tqdm import trange
import argparse

# Wrapper for Self Play
class SelfPlayPongWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SelfPlayPongWrapper, self).__init__(env)
        self.agent_id = 0  # Start with the perspective of player 0

    def reset(self):
        self.agent_id = np.random.choice([0, 1])  # Choose a random player perspective
        return super().reset()[self.agent_id]

    def step(self, action):
        actions = [action, action]  # Assuming the same action for both agents in self-play
        obs, reward, done, info = super().step(actions)
        return obs[self.agent_id], reward[self.agent_id], done[self.agent_id], info

    def render(self, mode='human'):
        return super().render(mode=mode)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on Pong Duel.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the agent.")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total timesteps for training.")
    args = parser.parse_args()

    # Create the environment
    env = SelfPlayPongWrapper(gym.make('ma_gym:PongDuel-v0'))

    # Create the agent
    model = DQN("MlpPolicy", env, learning_rate=args.learning_rate, exploration_initial_eps=1.0,
                exploration_final_eps=0.05, exploration_fraction=0.8, verbose=1, seed=42)

    # Train the agent and save it
    model.learn(total_timesteps=args.total_timesteps, log_interval=500)
    model.save("dqn_pongduel")

    # Remove to demonstrate saving and loading
    del model

    # Load trained agent
    model = DQN.load("dqn_pongduel")

    obs = env.reset()
    env.render()

    for episode in trange(500):
        terminated = False
        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            env.render()
            time.sleep(0.05)

        obs = env.reset()  # Explicitly reset the environment at the end of each episode
        print("Episode finished")