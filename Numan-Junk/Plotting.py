import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
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



def spike_density(data_array):
    


def IterateoverFiles(Mat_File_count,Plot=0,batch_size = 5, path = 'Unsupervised-Team-8/problem-2/data/test-data/',weight_decay_factor = 0.96):
    heartbeat_mixed = np.empty((2, Mat_File_count+1), dtype=object)
    for i in range(0, Mat_File_count+1, batch_size):
        if Plot == 1:
            fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
    
        for batch_idx, j in enumerate(range(i, min(i + batch_size, 153))):
            #path = 'Unsupervised-Team-8/problem-2/data/test-data/'
            mat_file = load_mat_file(path + f"{j:03}.mat")
            data00 = mat_file['val'].reshape(4, -1)

            # Perform ICA

            ica_components, x = Apply_ICA(data00)        

            # Perform FFT on ICA results
            fft_results, frequencies = Apply_FFT(ica_components) 

            # Analyze FFT to find max components
            max_index, max_index2 = Analyse_FFT_Result(frequencies,fft_results, weight_decay_factor)

            # Store max components in heartbeat_mixed array
            heartbeat_mixed[0, j] = fix_signal_shape(ica_components[max_index, :])
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

    return heartbeat_mixed
heartbeat_mixed = IterateoverFiles(152,0)
heartbeat_mixed = heartbeat_mixed.reshape(-1, 1)
#

"""
min_length = min([len(signal) for signal in heartbeat_mixed.reshape(-1)])
heartbeat_mixed = np.array([signal[:min_length] for signal in heartbeat_mixed.reshape(-1)])
print(heartbeat_mixed.shape)

covariance_matrix = np.cov(heartbeat_mixed.T)

# fig, ax = plt.subplots(nrows=1, ncols=1)
# plt.title('Covariance matrix')
# plt.axis('off')
# ax.imshow(covariance_matrix)
# plt.show()
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]
# plt.plot(np.linspace(1, len(eigenvalues), num=len(eigenvalues)), np.abs(eigenvalues))

# plt.ylabel('Eigenvalues size/strength')
# plt.xlabel('number of eigenvalue (sorted by strength)')
# plt.title('Eigenvalues sorted')
# plt.show()

total_variance_kept = eigenvalues[0:150].sum()
total_variance = eigenvalues.sum()

ratio_kept = total_variance_kept / total_variance

print(f'Variance kept: {total_variance_kept}, total variance: {total_variance}, ratio kept: {ratio_kept}')
pca = PCA(n_components=200)
X = pca.fit_transform(heartbeat_mixed)

eps = 2
min_samples = 2

cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)

# print(cluster.labels_)
print(f'DBSCAN found {len(np.unique(cluster.labels_))} clusters')
print(f'There is/are {len(np.argwhere(cluster.labels_ == -1))} outlier(s)')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster with a different color
unique_labels = np.unique(cluster.labels_)
for label in unique_labels:
    cluster_points = X[cluster.labels_ == label]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label}', edgecolor='k', s=100)

ax.set_title('DBSCAN Clusters')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
plt.show()
"""

