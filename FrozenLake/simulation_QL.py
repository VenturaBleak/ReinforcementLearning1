import numpy as np
import gymnasium as gym
import argparse
import pickle
import os

def simulate(environment, qtable, episodes, use_qtable=True):
    nb_success = 0

    for _ in range(episodes):
        state = environment.reset()
        state = state[0]
        done = False

        while not done:
            if use_qtable:
                action = np.argmax(qtable[state])
            else:
                action = environment.action_space.sample()

            new_state, reward, done, _, info = environment.step(action)
            print(f"Action: {action}, Reward: {reward}, New state: {new_state}")
            state = new_state
            nb_success += reward

    success_rate = (nb_success / episodes) * 100
    print(f"Success rate = {success_rate}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on FrozenLake.")
    parser.add_argument("--episodes", type=int, default=100, help="Total number of simulation episodes.")
    # use_qtable is a flag, so it doesn't need a value
    parser.add_argument("--use_qtable", action="store_true", help="Use trained Q-table for decisions. Default is random.",
                        default=True)
    parser.add_argument("--qtable_path", type=str, default=os.path.join("data","qtable.pkl"), help="Path to the saved Q-table.")
    parser.add_argument("--render_mode", action="store_true", default="human", help="Render the environment during simulation.")

    args = parser.parse_args()

    # Load the map from the file
    with open(os.path.join("data","frozen_lake_map.pkl"), "rb") as f:
        loaded_map = pickle.load(f)

    # Use the loaded map to create the environment
    env = gym.make('FrozenLake-v1', is_slippery = False, desc=loaded_map, render_mode=args.render_mode)

    if args.use_qtable:
        with open(args.qtable_path, 'rb') as f:
            qtable = pickle.load(f)
        print(f"Loaded Q-table from {args.qtable_path}")
    else:
        qtable = None
        print("Using random actions.")

    simulate(env, qtable, args.episodes, args.use_qtable)