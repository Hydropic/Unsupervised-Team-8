from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import matplotlib.pyplot as plt

unity_env = UnityEnvironment("./square_env/square.x86_64")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

####Self wrote ####
import Rand_loc_rot

# x=0 
# iter = 0
# while x == 0:
#     iter += 1
#     d_lr = 0
#     d_ud = -0.01
#     d_phi = 2*np.pi
#     action = [d_lr, d_ud, d_phi]
#     observation, *_ = env.step(action)
#     x = observation[2][0]
#     print(observation[2][0])
#     print(iter)

def create_test_data(x, i):
    iter = 0
    while iter <= i & i <= 9999:
        iter += 1
        action = Rand_loc_rot.Rand_loc_rot(x)
        action = np.array(action)
        observation, *_ = env.step(action)
        # plt.imshow(observation[0])
        # plt.imshow(observation[1])
        # plt.show()
        plt.imsave(f'results/Images/img{x}{iter:04}.jpeg', observation[0], cmap=None, format='jpeg')
        with open(f'results/Location_Rotation_Arrays/img{x}{iter:04}.txt', 'w') as f:
            f.write(str(observation[2]))
        
        #Reset Location 
        action = action * -1
        observation, *_ = env.step(action)
        ###### FOR DEBUGGING ######
        # plt.imsave(f'results/Images/img{x+5}{iter:04}.jpeg', observation[0], cmap=None, format='jpeg')
        # with open(f'results/Location_Rotation_Arrays/img{x+5}{iter:04}.txt', 'w') as f:
        #     f.write(str(observation[2]))
        ###########################
    if i >= 10000:
        print("Max Number of Iterations is 9999")
        return 0
    
    return 1