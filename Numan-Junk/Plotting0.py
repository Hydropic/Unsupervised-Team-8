import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
#import mne 

__path__ = 'Unsupervised-Team-8/problem-2/data/test-data/000'

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

def Analyse_FFT_Result(frequencies, fft_results,weight_decay_factor = 0.96):
    
    Sum_of_frequency = []
    for i in range(4):

        middle_index = len(frequencies) // 2
        low_freqs = np.abs(fft_results[i, :middle_index])
        high_freqs = np.abs(fft_results[i, middle_index:]) 
        weights = np.array([weight_decay_factor**i for i in range(len(low_freqs))])
        reversed_weights = np.array([weight_decay_factor**(i+len(low_freqs)) for i in range(len(high_freqs))])
        weighted_low_freqs = low_freqs * weights
        weighted_high_freqs = high_freqs * reversed_weights
        temp_sum = np.sum(weighted_low_freqs) + np.sum(weighted_high_freqs)
        Sum_of_frequency.append(temp_sum)
        # Add up the values and print
        print(f"Sum of frequency components for channel {i}: {Sum_of_frequency[i]}")


    temp_max = 0
    temp_max2 = 0
    for i in range(4):

        if (Sum_of_frequency[i] > temp_max):  
            #temp_max2 = temp_max
            temp_max = Sum_of_frequency[i]
            max_index = i

    for i in range(4):
        
        if (Sum_of_frequency[3-i] > temp_max2) & (3-i != max_index):
            temp_max2 = Sum_of_frequency[3-i]
            max_index2 = 3-i

    return max_index, max_index2


mat_file = load_mat_file(__path__)

data00 = mat_file['val'].reshape(4, -1)

print(data00.shape)

fig, ax = plt.subplots(4,1, figsize=(15,10))


x = np.linspace(0, data00.shape[1], data00.shape[1])

for i in range(0,4):
    ax[i].plot(x, data00[i, :], color='blue', label=f'electrode {i}')
    ax[i].set_xlabel('time (in [s])')
    ax[i].set_ylabel('heart rate in mV')

ax[0].set_title('Recroding from 000.mat (Test data)')


plt.tight_layout()
plt.show()

# Perform ICA
ica = FastICA(n_components=4)
ica_components = ica.fit_transform(data00.T).T

fig, ax = plt.subplots(4, 1, figsize=(15, 10))

x = np.linspace(0, data00.shape[1], data00.shape[1])

for i in range(0, 4):
    ax[i].plot(x, ica_components[i, :], color='blue', label=f'ICA component {i}')
    ax[i].set_xlabel('time (in [s])')
    ax[i].set_ylabel('ICA signal')

ax[0].set_title('ICA Components from 000.mat (Test data)')

plt.tight_layout()
plt.show()

ica_components, x = Apply_ICA(data00)       
# Perform FFT on ICA results
fft_results, frequencies = Apply_FFT(ica_components)
# Analyze FFT to find max components
max_index, max_index2 = Analyse_FFT_Result(frequencies,fft_results, weight_decay_factor=0.84)
# Store max components in heartbeat_mixed array





"""def IterateoverFiles(Mat_File_count,Plot=0,batch_size = 5, path = 'Unsupervised-Team-8/problem-2/data/test-data/',weight_decay_factor = 0.96):
    heartbeat_mixed = np.empty((2, Mat_File_count+1), dtype=object)
    for i in range(0, Mat_File_count+1, batch_size):
        if Plot == 1:
            fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
    
        for batch_idx, j in enumerate(range(i, min(i + batch_size, 153))):
            #path = 'Unsupervised-Team-8/problem-2/data/test-data/'
            mat_file = load_mat_file(path + f"{j:03}")
            data00 = mat_file['val'].reshape(4, -1)

            # Perform ICA

            ica_components, x = Apply_ICA(data00)        

            # Perform FFT on ICA results
            fft_results, frequencies = Apply_FFT(ica_components) heartbeat_mixed[0, j] = fix_signal_shape(ica_components[max_index, :])
heartbeat_mixed[1, j] = fix_signal_shape(ica_components[max_index2, :])[max_index, :]
            heartbeat_mixed[1, j] = fix_signal_shape(ica_components[max_index2, :])
        if Plot == 1:    
            # Plot ICA components with max_index and max_index2 for the current file
            ax[batch_idx, 0].plot(x, heartbeat_mixed[0, j], color='blue')
            ax[batch_idx, 0].set_title(f'ICA Max Component {batch_idx+1} for File {j:03}')
            ax[batch_idx, 0].set_xlabel('Time')
            ax[batch_idx, 0].set_ylabel(f'Amplitude({np.mean(heartbeat_mixed[0, j]):02})')

            ax[batch_idx, 1].plot(x, heartbeat_mixed[1, j], color='red')
            ax[batch_idx, 1].set_title(f'ICA 2nd Max Component {batch_idx+1} for File {j:03}')
            ax[batch_idx, 1].set_xlabel('Time')
            ax[batch_idx, 1].set_ylabel(f'Amplitude({np.mean(heartbeat_mixed[1, j]):02})')
        
            plt.tight_layout()
            plt.show()
            print(f"Batch starting at index {i}: Max index: {max_index}, 2nd Max index: {max_index2}")

    return heartbeat_mixed"""
#heartbeat_mixed = IterateoverFiles(152) 

from sklearn.decomposition import PCA

def pad_or_truncate(arrays, target_length):
    padded_arrays = []
    for arr in arrays:
        if len(arr) < target_length:
            # Pad with zeros
            padded_array = np.pad(arr, (0, target_length - len(arr)), 'constant')
        else:
            # Truncate
            padded_array = arr[:target_length]
        padded_arrays.append(padded_array)
    return np.array(padded_arrays)

def reshape_data(heartbeat_mixed):
    reshaped_data = []
    
    # Find the maximum signal length
    max_length = max(
        heartbeat_mixed[i, j].size
        for i in range(heartbeat_mixed.shape[0])
        for j in range(heartbeat_mixed.shape[1])
        if heartbeat_mixed[i, j] is not None
    )
    
    for i in range(heartbeat_mixed.shape[0]):
        for j in range(heartbeat_mixed.shape[1]):
            if heartbeat_mixed[i, j] is not None:
                signal = heartbeat_mixed[i, j]
                # Pad signal to match max_length
                padded_signal = np.pad(signal, (0, max_length - len(signal)))
                reshaped_data.append(padded_signal)

    return np.array(reshaped_data)

def apply_pca_with_elbow(heartbeat_mixed, max_components=306):
    # Reshape the data
    reshaped_data = reshape_data(heartbeat_mixed)

    # Normalize the data
    normalizer = StandardScaler()
    normalized_data = normalizer.fit_transform(reshaped_data)

    # Apply PCA for a range of components
    explained_variances = []
    for n_components in range(1, min(max_components, normalized_data.shape[1]) + 1):
        pca = PCA(n_components=n_components)
        pca.fit(normalized_data)
        explained_variances.append(sum(pca.explained_variance_ratio_))

    # Plot the elbow curve
    plt.figure(figsize=(24, 18))
    plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Elbow Method for Optimal PCA Components')
    plt.show()

    return explained_variances

#principal_components, explained_variance =  apply_pca_with_elbow(heartbeat_mixed)
