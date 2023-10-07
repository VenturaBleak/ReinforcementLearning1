from gym.envs.toy_text.frozen_lake import generate_random_map
import pickle
import os

seed = 42  # Any integer seed you choose
random_map = generate_random_map(size=5)

# make dir if not exists -> data
if not os.path.exists("data"):
    os.makedirs("data")

with open(os.path.join("data","frozen_lake_map.pkl"), "wb") as f:
    pickle.dump(random_map, f)