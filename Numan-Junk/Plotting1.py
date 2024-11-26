import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
#import mne 

__path__ = 'Unsupervised-Team-8/problem-2/data/test-data/011'

normalizer = StandardScaler()

def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)

    return mat

def flatten(arr,variable):
    data = arr[variable]
    data= data.flatten()
    return data

def get_mean_std(signal):
    mean = float(np.mean(signal))
    std_dev = float(np.std(signal))
    return mean, std_dev

def calculate_zScore(x,mean, std_dev):

    z_score = float((x-mean)/std_dev)
    return z_score

def Check_Spikes(a, threshold,mean, std_dev): # Sets 0 if under the threshold 
    temp_zScore = calculate_zScore(a,mean, std_dev)
    if abs(temp_zScore) >= threshold:
        return a
    elif temp_zScore < threshold:
        return 0

def fix_signal_shape(data_array):
    temp_data = []

    temp_mean, temp_std = get_mean_std(data_array)
    
    for i in range(data_array.size):
        temp_data.append(Check_Spikes(data_array[i], 2.0, temp_mean, temp_std))

    # Calculate the mean of the processed data
    channel_mean = np.mean(temp_data, dtype=np.float64)
    

    if channel_mean < 0:
        data_array = -data_array
    
    return data_array

def print_key_values(path):
    sample_1 = load_mat_file(path)

    keys = []
    values = []

    for key, value in sample_1.items():
        if not key.startswith('__'):  # Skip internal meta-keys
            keys.append(key)
            values.append(value.shape if hasattr(value, 'shape') else 'Not an array')

    # Create a DataFrame for better formatting
    df = pd.DataFrame({'Key': keys, 'Shape': values})

    # Print the table
    print(df)

def Apply_ICA(data):
    # Perform ICA
    ica = FastICA(n_components=4)
    ica_components = ica.fit_transform(data.T).T
    x = np.linspace(0, data.shape[1], data.shape[1])
    return ica_components, x

def Apply_FFT(ica_components):
    fft_results = np.fft.fft(ica_components, axis=1)
    n = ica_components.shape[1]
    frequencies = np.fft.fftfreq(n)
    return fft_results, frequencies
# Perform FFT on ICA results

def Find_2_min_Components(data_array):
        temp_min = float('inf')
        temp_min2 = float('inf')
        for i in range(len(data_array)):

            if (data_array[i] < temp_min):  
                #temp_max2 = temp_max
                temp_min = data_array[i]
                min_index = i

        for i in range(4):
            
            if (data_array[len(data_array)-1-i] < temp_min2) & (len(data_array)-1-i != min_index):
                temp_min2 = data_array[len(data_array)-1-i]
                min_index2 = len(data_array)-1-i

        return min_index, min_index2\
    

def Find_2_max_Components(data_array):
        temp_max = float('-inf')
        temp_max2 = float('-inf')
        for i in range(len(data_array)):

            if (data_array[i] > temp_max):  
                #temp_max2 = temp_max
                temp_max = data_array[i]
                max_index = i

        for i in range(4):
            
            if (data_array[len(data_array)-1-i] > temp_max2) & (len(data_array)-1-i != max_index):
                temp_max2 = data_array[len(data_array)-1-i]
                max_index2 = len(data_array)-1-i

        return max_index, max_index2

def Find_2_closest_Components(data_array):
    min_diff = float('inf')
    min_diff_index1 = 0
    min_diff_index2 = 0
    for i in range(4):
        for j in range(i+1, 4):
            diff = abs(data_array[i] - data_array[j])
            if diff < min_diff:
                min_diff = diff
                min_diff_index1 = i
                min_diff_index2 = j
    return min_diff_index1, min_diff_index2


def Analyse_FFT_Result(frequencies, fft_results,weight_decay_factor = 0.96):
   
    weight_decay_factor = 0.999
    weight_decay_first = 0.995
    divide_portion = 30
    exponential_multiplier = 2
    Sum_of_frequency = []
    for i in range(4):
        
        middle_index = len(frequencies) // 2
        low_freqs_first = np.abs(fft_results[i, :(middle_index//divide_portion)])
        low_freqs_second = np.abs(fft_results[i, middle_index//divide_portion:(divide_portion-1)*middle_index//divide_portion])
        high_freqs = np.abs(fft_results[i, middle_index:]) 
        weights_first = 2-exponential_multiplier*(np.array([weight_decay_first**(len(low_freqs_first)-i) for i in range(len(low_freqs_first))]))
        #weights_first = exponential_multiplier*(np.array([weight_decay_first**(i) for i in range(len(low_freqs_first))]))        
        weights_second = (-1)*(exponential_multiplier)*(np.array([weight_decay_factor**(i) for i in range(len(low_freqs_second))]))
        reversed_weights = exponential_multiplier*(np.array([weight_decay_factor**(i+len(low_freqs_first)+len(low_freqs_second)) for i in range(len(high_freqs))]))
        weighted_low_freqs_first = low_freqs_first *(weights_first)
        weighted_low_freqs_second = low_freqs_second *(weights_second)
        weighted_high_freqs = 0#high_freqs * reversed_weights*0
        temp_sum = np.sum(weighted_low_freqs_first)+ np.sum(weighted_low_freqs_second) + np.sum(weighted_high_freqs)
        Sum_of_frequency.append(temp_sum)
     
        # Add up the values and print
        print(f"Sum of frequency components for channel {i}: {Sum_of_frequency[i]}")


    min_index, min_index2 = Find_2_min_Components(Sum_of_frequency)
    max_index, max_index2 = Find_2_max_Components(Sum_of_frequency)

    return max_index, max_index2, min_index, min_index2







def IterateoverFiles(Mat_File_count,Plot=0,weight_decay_factor = 0.84,batch_size = 2, path = 'Unsupervised-Team-8/problem-2/data/test-data/'):
    heartbeat_mixed = np.empty((4, Mat_File_count+1), dtype=object)
    for i in range(1, Mat_File_count+1, batch_size):
        fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
    
        for batch_idx, j in enumerate(range(i, min(i + batch_size, Mat_File_count+1))):
            #path = 'Unsupervised-Team-8/problem-2/data/test-data/'
            mat_file = load_mat_file(path + f"{j:03}.mat")
            data00 = mat_file['val'].reshape(4, -1)

            # Perform ICA

            ica_components, x = Apply_ICA(data00)        

            # Perform FFT on ICA results
            fft_results, frequencies = Apply_FFT(ica_components) 

            # Analyze FFT to find max components
            max_index, max_index2, min_index, min_index2 = Analyse_FFT_Result(frequencies,fft_results, weight_decay_factor)

            # Store max components in heartbeat_mixed array
            heartbeat_mixed[0, j] = fix_signal_shape(ica_components[max_index, :])
            heartbeat_mixed[1, j] = fix_signal_shape(ica_components[max_index2, :])
            heartbeat_mixed[2, j] = fix_signal_shape(ica_components[min_index, :])
            heartbeat_mixed[3, j] = fix_signal_shape(ica_components[min_index2, :])
            
            # Plot ICA components with max_index and max_index2 for the current file
            ax[0, batch_idx].plot(x, heartbeat_mixed[0, j], color='blue')
            ax[0, batch_idx].set_title(f'ICA Max Component {batch_idx+1} for File {j:03}')
            ax[0, batch_idx].set_xlabel('Time')
            ax[0, batch_idx].set_ylabel(f'Amplitude({np.mean(heartbeat_mixed[0, j]):02})')

            ax[1, batch_idx].plot(x, heartbeat_mixed[1, j], color='blue')
            ax[1, batch_idx].set_title(f'ICA 2nd Max Component {batch_idx+1} for File {j:03}')
            ax[1, batch_idx].set_xlabel('Time')
            ax[1, batch_idx].set_ylabel(f'Amplitude({np.mean(heartbeat_mixed[1, j]):02})')


        if Plot == 1:
            plt.tight_layout()
            plt.show()
            print(f"Batch starting at index {i}: Max index: {max_index}, 2nd Max index: {max_index2}")

    return heartbeat_mixed
heartbeat_mixed = IterateoverFiles(25,1,weight_decay_factor = 0.7) 
print(heartbeat_mixed.shape)

