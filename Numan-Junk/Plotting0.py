import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

__path__ = 'Unsupervised-Team-8/problem-2/data/test-data/019'

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

fig, ax = plt.subplots(2, 1, figsize=(15, 10))

x = np.linspace(0, data00.shape[1], data00.shape[1])

short_array = [max_index, max_index2]
for i in range(0, 2):
    ax[i].plot(x, ica_components[short_array[i], :], color='blue', label=f'ICA component {i}')
    ax[i].set_xlabel('time (in [s])')
    ax[i].set_ylabel('ICA signal')

ax[0].set_title('ICA Components from 000.mat (Test data)')

#plt.tight_layout()
#plt.show()

fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i in range(0, 2):
    ax[i].plot(frequencies, np.abs(fft_results[short_array[i], :]), color='red', label=f'FFT of ICA component {i}')
    ax[i].set_xlabel('Frequency (Hz)')
    ax[i].set_ylabel('Amplitude')
    ax[i].set_xlim(0, np.max(frequencies)/2)  # Only plot positive frequencies

ax[0].set_title('FFT of ICA Components from 000.mat (Test data)')

plt.tight_layout()
plt.show()


