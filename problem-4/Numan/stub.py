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
    iter = 1    
    observation_list = []
    observation_list_2 = []
    while iter <= i & i <= 9999:
        iter += 1
        action = Rand_loc_rot.Rand_loc_rot(x)
        action = np.array(action)
        observation, *_ = env.step(action)
        observation_list.append(observation[0])
        observation_list_2.append(observation[2])
        # plt.imshow(observation[0])
        # plt.imshow(observation[1])
        # plt.show()
        #plt.imsave(f'results/Images/img{x}{iter:04}.jpeg', observation[0], cmap=None, format='jpeg')
        # with open(f'results/Location_Rotation_Arrays/img{x}{iter:04}.txt', 'w') as f:
        #     f.write(str(observation[2]))
        
        #Reset Location 
        if x!= 5:
            action = action * -1
            observation, *_ = env.step(action)
        ###### FOR DEBUGGING ######
        # plt.imsave(f'results/Images/img{x+6}{iter:04}.jpeg', observation[0], cmap=None, format='jpeg')
        # with open(f'results/Location_Rotation_Arrays/img{x+6}{iter:04}.txt', 'w') as f:
        #     f.write(str(observation[2]))
        ###########################
    array = np.array(observation_list, dtype=np.uint8)
    np.save(f'results/Images/img{x}{i:04}.npy', array, allow_pickle=True, fix_imports=True)
    array_2 = np.array(observation_list_2, dtype=np.uint8)
    np.save(f'results/Location_Rotation_Arrays/img{x}{i:04}.npy', array_2, allow_pickle=True, fix_imports=True)
    if i >= 10000:
        print("Max Number of Iterations is 9999")
        return 0
    
    return 1
