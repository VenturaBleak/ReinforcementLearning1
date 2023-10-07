import gym
import argparse
import torch
import pickle
import os
from model import QNetwork

def simulate(environment, model, episodes, use_model=True):
    nb_success = 0

    for _ in range(episodes):
        state = environment.reset()
        state = state[0]
        done = False

        while not done:
            state_tensor = torch.eye(environment.observation_space.n)[state].unsqueeze(0)
            if use_model:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()
            else:
                action = environment.action_space.sample()

            new_state, reward, done, _, info = environment.step(action)
            state = new_state
            nb_success += reward

    success_rate = (nb_success / episodes) * 100
    print(f"Success rate = {success_rate}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on FrozenLake.")
    parser.add_argument("--episodes", type=int, default=100, help="Total number of simulation episodes.")
    parser.add_argument("--use_model", action="store_true",
                        help="Use trained neural network for decisions. Default is random.", default=True)
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel_offline.pth"),
                        help="Path to the saved neural network model.")
    parser.add_argument("--render_mode", type=str, default="human",
                        help="Render mode for the environment: 'human' or None.")

    args = parser.parse_args()

    # Load the map from the file
    with open(os.path.join("data","frozen_lake_map.pkl"), "rb") as f:
        loaded_map = pickle.load(f)

    # Use the loaded map to create the environment
    env = gym.make('FrozenLake-v1', is_slippery = False, desc=loaded_map, render_mode=args.render_mode)

    if args.use_model:
        model = QNetwork(env.observation_space.n, env.action_space.n)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        print(f"Loaded model from {args.model_path}")
    else:
        model = None
        print("Using random actions.")

    simulate(env, model, args.episodes, args.use_model)