import gymnasium as gym
import argparse
import torch
import os

from model import QNetwork

def simulate(environment, model, episodes, use_model=True):
    total_reward = 0

    for _ in range(episodes):
        state = environment.reset()[0]
        done = False

        while not done:
            environment.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            if use_model:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()
            else:
                action = environment.action_space.sample()

            state, reward, done, _, info = environment.step(action)
            total_reward += reward

    average_reward = total_reward / episodes
    print(f"Average reward over {episodes} episodes: {average_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on CartPole.")
    parser.add_argument("--episodes", type=int, default=50, help="Total number of simulation episodes.")
    parser.add_argument("--use_model", action="store_true",
                        help="Use trained neural network for decisions. Default is random.", default=True)
    parser.add_argument("--model_path", type=str, default=os.path.join("data","DLmodel.pth"), help="Path to the saved neural network model.")

    args = parser.parse_args()

    env = gym.make('CartPole-v1', render_mode="human")

    if args.use_model:
        model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        print(f"Loaded model from {args.model_path}")
    else:
        model = None
        print("Using random actions.")

    simulate(env, model, args.episodes, args.use_model)