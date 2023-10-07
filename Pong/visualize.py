import time
from environment import PongDuel
import gymnasium

# Register the environment
gymnasium.register(
    id='PongDuel-v0',
    entry_point='environment:PongDuel'
)

env = gymnasium.make('PongDuel-v0')
done_n = [False for _ in range(env.unwrapped.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, _, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
    time.sleep(0.05)
env.close()