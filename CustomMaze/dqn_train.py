import gymnasium
from stable_baselines3 import DQN
from environment import MazeGameEnv
import pygame
from tqdm import trange
import argparse

# main
if __name__ == "__main__":

    # Register the environment
    gymnasium.register(
        id='MazeGame-v0',
        entry_point='environment:MazeGameEnv',
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on FrozenLake.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Total number of simulation episodes.")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total number of time.")
    args = parser.parse_args()

    # Create the environment
    env = gymnasium.make('MazeGame-v0')
    obs, _ = env.reset()
    # print('Initial observation:', obs)

    # Create the agent
    model = DQN("MlpPolicy", env, learning_rate=args.learning_rate, exploration_initial_eps=1.0, exploration_final_eps=0.05,
                exploration_fraction = 0.8,
                verbose=1, seed=42)

    # Train the agent and save it
    model.learn(total_timesteps=args.total_timesteps, log_interval=500)
    model.save("dqn_mazegame")

    # remove to demonstrate saving and loading
    del model

    # Load trained agent
    model = DQN.load("dqn_mazegame")

    obs, _ = env.reset()
    # print('Initial observation:', obs)
    env.render()

    for episode in trange(500):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            pygame.event.get()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # print('Reward:', reward)
            pygame.time.wait(40)

        obs, _ = env.reset() # Explicitly reset the environment at the end of each episode
        print("Episode finished")