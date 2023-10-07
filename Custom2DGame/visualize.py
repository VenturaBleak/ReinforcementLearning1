import gymnasium
import pygame
from environment import PlainFieldEnv  # Ensure this refers to the MazeGameEnv you've provided
import argparse
from tqdm import trange

# main
if __name__ == "__main__":

    # Register the environment
    gymnasium.register(
        id='PlainField-v0',
        entry_point='environment:PlainFieldEnv'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior in the MazeGame environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Total number of simulation episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gymnasium.make('PlainField-v0')
    obs, _ = env.reset()
    env.render()

    for episode in trange(args.episodes):
        terminated = False
        while not terminated:
            pygame.event.get()
            action = env.action_space.sample()  # Sample an action directly from the action space
            obs, reward, terminated, _, info = env.step(action)
            env.render()
            pygame.time.wait(50)

        obs, _ = env.reset()  # Explicitly reset the environment at the end of each episode
        print("Episode finished")