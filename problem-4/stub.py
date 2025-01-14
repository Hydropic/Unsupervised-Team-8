from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import matplotlib.pyplot as plt

unity_env = UnityEnvironment("./square_env/SquareRoom.exe")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

