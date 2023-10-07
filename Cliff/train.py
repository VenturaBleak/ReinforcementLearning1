import gymnasium
from stable_baselines3 import DQN
from tqdm import trange
import argparse
import time

# main
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate the agent's behavior on CliffWalk.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Total number of simulation episodes.")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total number of time.")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Total number of simulation episodes.")
    parser.add_argument("-policy", type=str, default="True", help="Policy to use for action selection (random or policy).")
    args = parser.parse_args()

    # Create the environment
    env = gymnasium.make('CliffWalking-v0')
    obs, _ = env.reset()

    # Create the agent
    model = DQN("MlpPolicy", env, learning_rate=args.learning_rate, exploration_initial_eps=1.0, exploration_final_eps=0.2,
                exploration_fraction = 0.8,
                verbose=1, seed=42)

    # Train the agent and save it
    model.learn(total_timesteps=args.total_timesteps, log_interval=100)
    model.save("model")

    # remove to demonstrate saving and loading
    del model

    # Load trained agent
    model = DQN.load("model")

    # Simulate the agent's behavior on the environment
    env = gymnasium.make('CliffWalking-v0', render_mode='human')
    nb_success = 0
    obs, _ = env.reset()
    env.render()

    for episode in trange(args.eval_episodes):
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # if policy == True, then sample action from policy, else sample random action
            if args.policy:
                action, _states = model.predict(obs, deterministic=True)
                action = int(action)  # Convert numpy array to integer
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            # print reward and take next step in the environment
            print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

            env.render()

            nb_success += reward
            time.sleep(0.1)

        obs, _ = env.reset() # Explicitly reset the environment at the end of each episode
        print("Episode finished")

    success_rate = (nb_success / args.eval_episodes) * 100
    print(f"Success rate = {success_rate}%")