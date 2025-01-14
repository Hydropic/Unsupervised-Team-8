from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

# This initializes the environment to interact with
unity_env = UnityEnvironment("./square_env/SquareRoom.exe")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

# This resets the environment to an initial state and provides an initial observation
# - observations are three numpy arrays 
obs_ego, obs_top, vectorial = env.reset()
# obs_ego is the observation that the robot-camera provides (your data)
# obs_top is a topdown view (debug only)
# vectorial contains non-visual signals:
# -> true/false if wall was hit (can be used)
# -> x-position, y-position, rotation angle (ground truth, debug only)

time_between = 0
def init():
    return [im_left, im_right]

def animate(i):
    # Take a step in the environment providing 3d vector (x-movement, y-movement, rotation angle)
    global last_time
    current_time = time.time()
    time_between = current_time - last_time
    print(f"Current FPS {1/time_between:.2f}")
    last_time = current_time
    step_output = env.step(np.random.uniform(-0.5, 0.5, 3))
    obs_ego, obs_top, vectorial = step_output[0]
    im_left.set_array(obs_ego)
    im_right.set_array(obs_top)
    return [im_left, im_right]

fig, ax = plt.subplots(1, 2)
im_left = ax[0].imshow(obs_ego, interpolation='none')
im_right = ax[1].imshow(obs_top, interpolation='none')

last_time = time.time()
anim = animation.FuncAnimation(fig,
                               animate,
                            init_func=init,
                               frames = 5,
                               interval=1)

plt.show()

