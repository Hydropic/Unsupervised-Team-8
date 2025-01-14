from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import time

unity_env = UnityEnvironment("./square_env/SquareRoom.exe")
env = UnityToGymWrapper(unity_env)
_ = env.reset()

start = time.time()
n_frames = 250
for _ in range(n_frames):
    _ = env.step(np.random.uniform(-0.5, 0.5, 3))
elapsed = time.time() - start
print(f"Took {elapsed:.2f}s")
print(f"This corresponds to {n_frames/elapsed:.2f} fps.")


