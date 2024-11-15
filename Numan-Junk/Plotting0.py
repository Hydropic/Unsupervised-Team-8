import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import mne 

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


sample_1 = load_mat_file(__path__)

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

data = sample_1['val']

channel_2 =  data[0, 1, :]
channel_1 =  data[0, 0, :]
channel_3 =  data[0, 2, :]
channel_4 =  data[0, 3, :]



#plt.figure(figsize=(10, 8))
for i, channel in enumerate([channel_1, channel_2, channel_3, channel_4], start=1):
    plt.subplot(4, 1, i)
    plt.plot(channel)
    plt.title(f"Channel {i}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

#plt.tight_layout()
#plt.show()

mat_file = load_mat_file(__path__)

data00 = mat_file['val'].reshape(4, -1)

print(data00.shape)

# Perform ICA
ica = FastICA(n_components=4)
ica_components = ica.fit_transform(data00.T).T
#ica_components = normalizer.fit_transform(ica_components)


# Perform FFT on ICA results
fft_results = np.fft.fft(ica_components, axis=1)

#fft_results = normalizer.fit_transform(fft_results.reshape(-1,1)).flatten()
# Compute frequencies


# Plot the FFT results

from scipy.signal import butter, filtfilt

def low_pass_filter(data, cutoff=20, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def high_pass_filter(data, cutoff=1, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def Analyse_FFT_Result(frequencies, fft_results):
    weight_decay_factor = 0.96
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

    #print(len(frequencies))

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
#max_index, max_index2 = Analyse_FFT_Result(frequencies)
#print(f"Max index: {max_index}, 2nd Max index: {max_index2}")

batch_size = 5

# Initialize heartbeat_mixed to store two components (max_index and max_index2) for each file
# with each time series component having the length of the original data (1500)
heartbeat_mixed = np.empty((2, 153), dtype=object)

for i in range(1, 26, batch_size):
    fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
    
    for batch_idx, j in enumerate(range(i, min(i + batch_size, 153))):
        path = 'Unsupervised-Team-8/problem-2/data/real-data/'
        mat_file = load_mat_file(path + f"a{j:02}m.mat")
        data00 = mat_file['val'].reshape(4, -1)

        data00 = low_pass_filter(data00)
        data00 = high_pass_filter(data00)

        # Perform ICA
        ica = FastICA(n_components=4)
        ica_components = ica.fit_transform(data00.T).T
        x = np.linspace(0, data00.shape[1], data00.shape[1])
        



        # Perform FFT on ICA results
        fft_results = np.fft.fft(ica_components, axis=1)
        n = ica_components.shape[1]
        frequencies = np.fft.fftfreq(n)

        # Analyze FFT to find max components
        max_index, max_index2 = Analyse_FFT_Result(frequencies,fft_results)

        # Store max components in heartbeat_mixed array
        heartbeat_mixed[0, j] = fix_signal_shape(ica_components[max_index, :])
        heartbeat_mixed[1, j] = fix_signal_shape(ica_components[max_index2, :])

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


#plt.show()
