import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd # To check the Keys etc.
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
import os
import scipy.signal
import scipy.stats
from scipy.fft import fft, fftfreq
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import math
from sklearn.mixture import GaussianMixture

#import mne 

__path__ = 'Unsupervised-Team-8/problem-2/data/test-data/011'

normalizer_standart = StandardScaler()
normalizer_minmax = MinMaxScaler()

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
    ica = FastICA(
    n_components=4,        # Number of components to extract        # Perform whitening
    fun='logcosh',         # Contrast function for independence
    max_iter=5000,          # Maximum number of iterations
    tol=1e-6,              # Tolerance for convergence
    random_state=2        # Ensure reproducibility
)
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


level =2
def peak_density(data_array):
    peak_count, _ = find_peaks(data_array, height=level)  # Adjust the 'height' parameter to change sensitivity
    peak_count = len(peak_count)
    peak_density = peak_count / len(data_array)
    return peak_density

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
#heartbeat_mixed = heartbeat_mixed.reshape(-1, 1)
unmixed_heartbeat = []
skipped_files = []
for i in range(153):
    dens1 = peak_density(heartbeat_mixed[0][i])
    dens2 = peak_density(heartbeat_mixed[1][i])
    if dens1 > dens2:
        unmixed_heartbeat.append(heartbeat_mixed[0][i])
        skipped_files.append(heartbeat_mixed[1][i])
    else:
        unmixed_heartbeat.append(heartbeat_mixed[1][i])
        skipped_files.append(heartbeat_mixed[0][i])
        
# print(heartbeat_mixed.shape)
# print(heartbeat_mixed[0][0].shape)
#
batch_size = 5
Mat_File_count = 152
path = 'Unsupervised-Team-8/problem-2/data/test-data/'
def plot_last():
    for i in range(0, Mat_File_count+1, batch_size):

        fig, ax = plt.subplots(batch_size, 2, figsize=(15, 20))
        
        for batch_idx, j in enumerate(range(i, min(i + batch_size, 153))):
            #path = 'Unsupervised-Team-8/problem-2/data/test-data/'
            mat_file = load_mat_file(path + f"{j:03}.mat")
            data00 = mat_file['val'].reshape(4, -1)
            peak_unmixed = find_peaks(unmixed_heartbeat[j],height=level)[0]
            peak_mixed = find_peaks(skipped_files[j],height=level)[0]
        
    
                # Plot ICA components with max_index and max_index2 for the current file
            ax[batch_idx, 0].plot(unmixed_heartbeat[j], color='blue')
            ax[batch_idx, 0].scatter(peak_unmixed, unmixed_heartbeat[j][peak_unmixed], color='green')
            ax[batch_idx, 0].set_title(f'ICA Max Component {batch_idx+1} for File {j:03}')
            ax[batch_idx, 0].set_xlabel('Time')
            ax[batch_idx, 0].set_ylabel(f'Amplitude')

            ax[batch_idx, 1].plot(skipped_files[j], color='red')
            ax[batch_idx, 1].scatter(peak_mixed, skipped_files[j][peak_mixed], color='green')
            ax[batch_idx, 1].set_title(f'ICA 2nd Max Component {batch_idx+1} for File {j:03}')
            ax[batch_idx, 1].set_xlabel('Time')
            ax[batch_idx, 1].set_ylabel(f'Amplitude')
            
        plt.tight_layout()
        plt.show()


#plot_last()

min_length = min([len(signal) for signal in unmixed_heartbeat])
unmixed_heartbeat = np.array([signal[:min_length] for signal in unmixed_heartbeat])
print(unmixed_heartbeat.shape)

covariance_matrix = np.cov(unmixed_heartbeat.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]
total_variance = eigenvalues.sum()

normalizer_minmax = MinMaxScaler()
normalized_data = normalizer_minmax.fit_transform(unmixed_heartbeat)

input_dim = min_length  # Number of features

shell3 = 16
shell2 = 8
shell1 = 4
core = 2

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(shell3, activation='selu')(input_layer)
encoded = layers.Dense(shell2, activation='selu')(encoded)
encoded = layers.Dense(shell1, activation='selu')(encoded)

# Latent Space
latent_space = layers.Dense(core, activation='selu')(encoded)
#drop it to 2
# Decoder
decoded = layers.Dense(shell1, activation='selu')(latent_space)
decoded = layers.Dense(shell2, activation='selu')(decoded)
decoded = layers.Dense(shell3, activation='selu')(decoded)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

# Model
autoencoder = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
autoencoder.compile(optimizer='rmsprop', loss='mse')


X_train, X_val = train_test_split(normalized_data, test_size=0.2, random_state=2)

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,  # Input and target are the same for unsupervised learning
    epochs=15,
    batch_size=8,
    validation_data=(X_val, X_val),
    shuffle=False
)


reconstructed = autoencoder.predict(X_val)
reconstruction_error = np.mean((X_val - reconstructed) ** 2, axis=1)

# Flag potential anomalies
threshold = np.percentile(reconstruction_error, 99)  # Define threshold (e.g., 95th percentile)
anomalies = reconstruction_error > threshold

print(f"Threshold for anomaly detection: {threshold}")
print(f"Number of anomalies detected: {len(anomalies)}")

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.axvline(x=threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()


reconstructed = autoencoder.predict(normalized_data)
reconstruction_error = np.mean((normalized_data - reconstructed) ** 2, axis=1)

# Define the anomaly threshold
threshold = np.percentile(reconstruction_error, 95)

# Get indices of anomalies
anomalies = np.where(reconstruction_error > threshold)[0]

# Plot some anomalies
for i, idx in enumerate(anomalies[:5]):  # Plot the first 5 anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(unmixed_heartbeat[idx], label=f"Anomalous Heartbeat {i+1}")
    plt.title(f"Anomalous Heartbeat {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

latent_model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_3').output)

# Get latent representations
latent_features = latent_model.predict(normalized_data)

# Compute variance
latent_variance = np.var(latent_features, axis=0)
#total_variance = np.sum(latent_variance)
cumulative_variance = np.cumsum(latent_variance) #/ total_variance

total_variance = eigenvalues.sum()
# Plot cumulative variance
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_variance, marker='o')
# plt.title("Cumulative Variance Explained by Latent Features")
# plt.xlabel("Number of Latent Features")
# plt.ylabel("Cumulative Variance Explained")
# plt.grid()
# plt.show()


def extract_features(signal, sampling_rate):
    features = {}

    # 1. Rhythm Features
    peaks, _ = scipy.signal.find_peaks(signal, distance=sampling_rate * 0.6)  # Adjust distance based on expected heart rate
    ibi = np.diff(peaks) / sampling_rate  # Inter-beat intervals in seconds
    features['mean_ibi'] = np.mean(ibi) if len(ibi) > 0 else 0
    features['sdnn'] = np.std(ibi) if len(ibi) > 0 else 0

    # 2. Time-Domain Features
    features['mean'] = np.mean(signal)
    features['median'] = np.median(signal)
    features['variance'] = np.var(signal)
    features['std_dev'] = np.std(signal)
    features['skewness'] = scipy.stats.skew(signal)
    features['kurtosis'] = scipy.stats.kurtosis(signal)
    features['peak_to_peak'] = np.ptp(signal)

    # 3. Frequency-Domain Features
    n = len(signal)
    freqs = fftfreq(n, 1 / sampling_rate)
    fft_values = fft(signal)
    power = np.abs(fft_values) ** 2

    # Bandwidth: Range of significant frequencies
    significant_freqs = freqs[power > np.max(power) * 0.1]  # Adjust threshold as needed
    features['bandwidth'] = (np.max(significant_freqs) - np.min(significant_freqs)) if len(significant_freqs) > 0 else 0

    # Spectral Entropy
    power_normalized = power / np.sum(power)
    spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-12))  # Avoid log(0)
    features['spectral_entropy'] = spectral_entropy

    return features


def extract_features_for_all(data, sampling_rate):
    all_features = []
    for signal in data:
        features = extract_features(signal, sampling_rate)
        all_features.append(list(features.values()))  # Convert feature dictionary to a list
    return np.array(all_features)

# Plotting the elbow curve

def compare_variance(raw_data, features, latent_features):
    # Standardize raw data and features
    scaler = StandardScaler()
    scaled_raw_data = scaler.fit_transform(raw_data)
    scaled_features = scaler.fit_transform(features)

    # PCA for raw data
    pca_raw = PCA()
    pca_raw.fit(scaled_raw_data)
    explained_variance_raw = np.cumsum(pca_raw.explained_variance_ratio_)
    #explained_variance_raw = np.cumsum(pca_raw)/ total_variance

    # PCA for feature-extracted data
    pca_features = PCA()
    pca_features.fit(scaled_features)
    explained_variance_features = np.cumsum(pca_features.explained_variance_ratio_)
    #explained_variance_features = np.cumsum(pca_features)/ total_variance
    
    latent_features = scaler.fit_transform(latent_features)
    pca_latent = PCA()
    pca_latent.fit(latent_features)
    cumulative_variance = np.cumsum(pca_latent.explained_variance_ratio_)

    cumulative_variance0 = np.sum(latent_variance)
    #cumulative_variance = np.cumsum(pca_latent)/ total_variance
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_raw) + 1), explained_variance_raw, label='Raw Data', marker='o')
    plt.plot(range(1, len(explained_variance_features) + 1), explained_variance_features, label='Feature-Extracted Data', marker='x')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label='Latent Features', marker='o')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid()
    plt.show()
    return explained_variance_raw/ total_variance, explained_variance_features/ total_variance , cumulative_variance / total_variance

def hierarchical_clustering(latent_features):
    # Step 1: Generate the linkage matrix
    linkage_matrix = sch.linkage(latent_features, method='ward')  # Ward's method minimizes variance within clusters
    
    # Step 2: Plot the dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

    # Step 3: Choose the number of clusters (e.g., based on dendrogram)
    n_clusters = 2  # Adjust this based on your dendrogram
    
    # Step 4: Perform Agglomerative Clustering
    hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hc_model.fit_predict(latent_features)
    # Step 5: Plot the cluster assignments in 3D using the highest-variance latent components
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(latent_features[:,0], latent_features[:, 1], c=cluster_labels)
    ax.set_xlabel('Latent Component 1')
    ax.set_ylabel('Latent Component 2')

    ax.set_title("Hierarchical Clustering Results")
    plt.show()
    
    return cluster_labels



def gaussian_mixture_clustering(latent_features, n_components=2):

    # Step 1: Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(latent_features)
    cluster_labels = gmm.predict(latent_features)

    # Step 2: Plot GMM Clustering Results
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_features[:, 0], latent_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster Label')
    plt.xlabel('Latent Component 1')
    plt.ylabel('Latent Component 2')
    plt.title('Gaussian Mixture Clustering Results')
    plt.show()

    return cluster_labels



if __name__ == "__main__":
    # Simulated raw data (replace with your actual raw heartbeat recordings)
    sampling_rate = 360  # Hz
    raw_data = unmixed_heartbeat
    raw_data = np.array(raw_data)  # Shape: (n_samples, time_points)

    # Simulated feature-extracted data (use actual features from the previous process)
    features = extract_features_for_all(raw_data, sampling_rate)  # Replace with actual feature extraction

    # Compare variance
    raw_variance, feature_variance, latent_variance = compare_variance(raw_data, features, latent_features)

    scaler = StandardScaler()
    latent_features_scaled = scaler.fit_transform(latent_features)    
    cluster_assignments = gaussian_mixture_clustering(latent_features_scaled)

    
    # Output cluster assignments
    print("Cluster Assignments for Each Data Point:", cluster_assignments)

#print(f"Raw variance {raw_variance}, Feature variance {feature_variance},AutoEncoder Variance {latent_variance}, Total Variance {total_variance}")

"""covariance_matrix = np.cov(unmixed_heartbeat.T)

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.title('Covariance matrix')
plt.axis('off')
ax.imshow(covariance_matrix)
plt.show()
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]
plt.plot(np.linspace(1, len(eigenvalues), num=len(eigenvalues)), np.abs(eigenvalues))

plt.ylabel('Eigenvalues size/strength')
plt.xlabel('number of eigenvalue (sorted by strength)')
plt.show()
plt.title('Eigenvalues sorted')
"""


