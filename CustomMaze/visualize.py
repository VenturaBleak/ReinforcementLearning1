import gymnasium
import pygame
from environment import MazeGameEnv
import argparse
from tqdm import trange

# main
if __name__ == "__main__":

    # Register the environment
    gymnasium.register(
        id='MazeGame-v0',
        entry_point='environment:MazeGameEnv'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on FrozenLake.")
    parser.add_argument("--episodes", type=int, default=100, help="Total number of simulation episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gymnasium.make('MazeGame-v0')
    obs, _ = env.reset()
    # print('Initial observation:', obs)
    env.render()

    for episode in trange(args.episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            pygame.event.get()
            action = env.random_valid_action()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # print('Reward:', reward)
            pygame.time.wait(40)

            # Added for debugging purposes
            # print("Done status:", done)

        obs, _ = env.reset() # Explicitly reset the environment at the end of each episode
        print("Episode finished")