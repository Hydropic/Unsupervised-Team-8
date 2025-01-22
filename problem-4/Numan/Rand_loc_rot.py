import numpy as np
import math
import random
rng = np.random.default_rng()
rng_phi = np.random.default_rng()
print(random.randint(0, 1))
print(random.random())

def Rand_loc_rot(x):
    north_south = random.randint(0, 1)
    east_west = random.randint(0, 1)
    rot = random.random()*2*np.pi
    d_lr = random.randint(0, 750)*0.01
    d_ud = random.randint(0, 750)*0.01
    if north_south == 0:
        d_ud = -d_ud
    if east_west == 0:
        d_lr = -d_lr

    if x == 0:
        pass
    elif x == 1: #East
        d_lr = abs(d_lr)
        d_ud = 0
    elif x == 2: #North
        d_ud = abs(d_ud)
        d_lr = 0
    elif x == 3: #West
        d_lr = abs(d_lr)*-1
        d_ud = 0
    elif x == 4: #South
        d_ud = abs(d_ud)*-1
        d_lr = 0
    elif x == 5: #Random
        d_lr = d_lr*0.01
        d_ud = d_ud*0.01
        rot = rot*0.1
    action = [d_lr, d_ud, rot]
    return action
