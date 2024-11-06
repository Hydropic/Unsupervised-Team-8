import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

__path__ = 'Unsupervised-Team-8/problem-2/data/test-data/011'

normalizer = StandardScaler()

def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)

    return mat

def flatten(arr,variable):
    data = arr[variable]
    data= data.flatten()
    return data

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


# Perform FFT on ICA results
fft_results = np.fft.fft(ica_components, axis=1)

#fft_results = normalizer.fit_transform(fft_results.reshape(-1,1)).flatten()
# Compute frequencies
n = ica_components.shape[1]
frequencies = np.fft.fftfreq(n)

# Plot the FFT results


def Analyse_FFT_Result():
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
max_index, max_index2 = Analyse_FFT_Result()
print(f"Max index: {max_index}, 2nd Max index: {max_index2}")

batch_size = 5

# Initialize heartbeat_mixed to store two components (max_index and max_index2) for each file
# with each time series component having the length of the original data (1500)
heartbeat_mixed = np.empty((2, 153), dtype=object)

for i in range(0, 153, batch_size):
    fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
    
    for batch_idx, j in enumerate(range(i, min(i + batch_size, 153))):
        path = 'Unsupervised-Team-8/problem-2/data/test-data/'
        mat_file = load_mat_file(path + f"{j:03}")
        data00 = mat_file['val'].reshape(4, -1)

        # Perform ICA
        ica = FastICA(n_components=4)
        ica_components = ica.fit_transform(data00.T).T
        x = np.linspace(0, data00.shape[1], data00.shape[1])
        
        # Perform FFT on ICA results
        fft_results = np.fft.fft(ica_components, axis=1)

        # Analyze FFT to find max components
        max_index, max_index2 = Analyse_FFT_Result()

        # Store max components in heartbeat_mixed array
        heartbeat_mixed[0, j] = ica_components[max_index, :]
        heartbeat_mixed[1, j] = ica_components[max_index2, :]

        # Plot ICA components with max_index and max_index2 for the current file
        ax[batch_idx, 0].plot(x, heartbeat_mixed[0, j], color='blue')
        ax[batch_idx, 0].set_title(f'ICA Max Component {batch_idx+1} for File {j:03}')
        ax[batch_idx, 0].set_xlabel('Time')
        ax[batch_idx, 0].set_ylabel('Amplitude')

        ax[batch_idx, 1].plot(x, heartbeat_mixed[1, j], color='red')
        ax[batch_idx, 1].set_title(f'ICA 2nd Max Component {batch_idx+1} for File {j:03}')
        ax[batch_idx, 1].set_xlabel('Time')
        ax[batch_idx, 1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    print(f"Batch starting at index {i}: Max index: {max_index}, 2nd Max index: {max_index2}")


#plt.show()
