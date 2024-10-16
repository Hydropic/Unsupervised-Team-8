import scipy.io
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd # To check the Keys etc.

############### Variables - Paths - Arrays ###############

__path__ = 'Datasets/sample_1' #Update for yourself



##########################################################
########## Utilities For Single Point Iteration ##########

def Print_Keys_and_Values(arr):   #FROMGPT
    keys = []
    values = []
    for key, value in arr.items():
        if not key.startswith('__'):  # Skip internal meta-keys
            keys.append(key)
            values.append(value.shape if hasattr(value, 'shape') else 'Not an array')
    df = pd.DataFrame({'Key': keys, 'Shape': values})
    return df

def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)

    return mat

def flatten(arr,variable):
    data = arr[variable]
    data= data.flatten()
    return data

def calculate_zScore(x,arr):
    mean = float(np.mean(arr))
    std_dev = float(np.std(arr))
    z_score = float((x-mean)/std_dev)
    return z_score

##########################################################
################### Main Functions #######################

def Check_Spikes(a, dataset,threshold): # Sets 0 if under the threshold 
    temp_zScore = calculate_zScore(a,dataset)
    if abs(temp_zScore) >= threshold:
        return a
    elif temp_zScore < threshold:
        return 0
    
def Create_spikeArray(dataset, interval_start, interval_length, update_interval):
    temp_array = []
    for i in range(len(dataset[interval_start:interval_length+interval_start])):
        temp_array.append(Check_Spikes(dataset[i], dataset,2))
    
        if i%update_interval == 0:
            print(f"Creating Spike Array : {int(i/update_interval)} of {int(len(dataset[interval_start:interval_length+interval_start])/update_interval)}")

    data_array = np.array(temp_array).reshape(1, -1)
    data_array = data_array.flatten()
    return data_array

def plot(result_array):
    plt.plot(result_array)
    plt.xlabel('Sample Index')  # Label for the x-axis
    plt.ylabel('Value')   
    plt.show()

############################################################

mat=load_mat_file(__path__)

print(Print_Keys_and_Values(mat))

data=flatten(mat,'data')

result = Create_spikeArray(data, 0, 5000, 1000)

plot(result)