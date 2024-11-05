import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import ParameterGrid

__path__ = 'Unsupervised-Team-8/Datasets/sample_1.mat' 
def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)

    return mat

def flatten(arr,variable):
    data = arr[variable]
    data= data.flatten()
    return data

low_bound = 0
up_bound = 10000

mat_file = load_mat_file(__path__)
eeg_signal_total = flatten(mat_file,'data')
eeg_signal_total = (eeg_signal_total)
eeg_signal = eeg_signal_total[low_bound:up_bound]
eeg_signal = (eeg_signal)
### TRY ABS FOR BETTER ####

################ Creating Bandpass Filter #####################
from scipy.signal import butter, filtfilt

# Bandpass filter function
from scipy.signal import butter, filtfilt

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, samplingInterval, order=4):
    # Calculate the sampling frequency (fs) from samplingInterval
    fs = 1 / samplingInterval  # fs is the inverse of sampling interval
    
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize lowcut
    high = highcut / nyquist  # Normalize highcut
    
    # Ensure low < high and within 0 < Wn < 1
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError("Critical frequencies must be between 0 and 1 after normalization.")
    
    # Design Butterworth filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter
    y = filtfilt(b, a, data)
    return y

# Example usage: assuming the sampling interval is in seconds (e.g., 0.001 for 1 kHz sampling rate)
samplingInterval = 0.001  # Adjust this based on your actual data
lowcut = 0.1  # Desired low cut frequency in Hz
highcut = 100  # Desired high cut frequency in Hz

# Filter the data
#eeg_signal = bandpass_filter(eeg_signal, 0.01, 100, 0.001)

mean = float(np.mean(eeg_signal_total))
std_dev = float(np.std(eeg_signal_total))

def calculate_zScore(x,mean, std_dev):

    z_score = float((x-mean)/std_dev)
    return z_score

def Check_Spikes(a, threshold): # Sets 0 if under the threshold 
    temp_zScore = calculate_zScore(a,mean, std_dev)
    if abs(temp_zScore) >= threshold:
        return a
    elif temp_zScore < threshold:
        return 0

def Create_spikeArray(dataset, interval_start, interval_length, update_interval):
    temp_array = []
    start_time = time.time() #Measure the Performance
    for i in range(len(dataset[interval_start:interval_length+interval_start])):         
        temp_array.append(Check_Spikes(dataset[i+interval_start], 0.0))
    
        if i%update_interval == 0:
            print(f"Creating Spike Array : {int(i/update_interval)} of {int(len(dataset[interval_start:interval_length+interval_start])/update_interval)} in {time.time()-start_time} seconds")
            start_time = time.time() # Measure the Performance

    data_array = np.array(temp_array).reshape(1, -1)
    data_array = data_array.flatten()
    return data_array
eeg_signal = Create_spikeArray(eeg_signal_total,low_bound,up_bound-low_bound,1000)
###############################################################




# Assuming `eeg_signal` is your EEG data (1D array)
window_size = 200  # Define window size
stride = 50  # Define the stride for overlapping windows

# Normalize the data
scaler = StandardScaler()
eeg_signal = scaler.fit_transform(eeg_signal.reshape(-1, 1)).flatten()

# Create windows from the EEG signal
def create_windows(data, window_size, stride):
    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

from sklearn.neighbors import LocalOutlierFactor
#Reshape data for LOF
data_points = eeg_signal.reshape(-1, 1)

# Fit LOF with a number of neighbors


# Detect spikes (where LOF score is negative)
#


from sklearn.neighbors import LocalOutlierFactor
import numpy as np
"""# Reshape data for LOF
data_points = eeg_signal.reshape(-1, 1)

# Fit LOF as usual
lof = LocalOutlierFactor(n_neighbors=60, metric='minkowski')
outlier_scores = -lof.fit_predict(data_points)  # LOF scores (higher means more anomalous)

# Apply weighting to focus more on past neighbors (exponentially decay the scores)
weight_decay_factor = 0.9
weights = np.array([weight_decay_factor**i for i in range(len(outlier_scores))])

# Weighted LOF scores
weighted_outlier_scores = outlier_scores * weights

# Define a threshold for spikes based on weighted scores
threshold = np.percentile(weighted_outlier_scores, 80)
spike_indices = np.where(weighted_outlier_scores > threshold)[0]"""


param_grid = {
    "n_neighbors": list(range(1, 801, 10)),
    "confidence_threshold" : [0,1,2,3,4,5,6, 7, 8, 9, 10,11,12,13,14,15]
}

best_score = float('-inf')
best_params = None

"""# Perform grid search
for params in ParameterGrid(param_grid):
    lof = LocalOutlierFactor(n_neighbors=params['n_neighbors'])
    outlier_scores = lof.fit_predict(data_points)
    spike_indices = np.where(outlier_scores == -1)[0]
    confidence_threshold = params['confidence_threshold']

    # Group consecutive indices
    groups = []
    current_group = [spike_indices[0]]

    for i in range(1, len(spike_indices)):
        if spike_indices[i] == spike_indices[i - 1] + 1:
            current_group.append(spike_indices[i])
        else:
            groups.append(current_group)
            current_group = [spike_indices[i]]
    groups.append(current_group)

    # Filter groups based on confidence and get middle indices
    filtered_middle_indices = []

    for group in groups:
        confidence = len(group)
        if confidence >= confidence_threshold:
            middle_index = group[len(group) // 2]
            filtered_middle_indices.append(middle_index)

    spike_times = mat_file['spike_times'][0][0][0]

    # See how many spikes are detected also accept detected spikes that are near the ground truth
    score = 0
    for spike in spike_times:
        for detected_spike in filtered_middle_indices:
            if abs(spike - detected_spike) <= 50:
                print(abs(spike-detected_spike))
                score += 1
                break

    if score > best_score:
        best_score = score
        best_params = params
"""
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

lof = LocalOutlierFactor(n_neighbors=50)
outlier_scores = lof.fit_predict(data_points)

"""# Iterate through the outliers array
for i in range(len(outlier_scores)):
    if outlier_scores[i] == -1:  # Check if current point is an outlier
        # Find the index of the max value in the next 40 points of the signal
        if i + 40 < len(eeg_signal):
            temp_outlier = np.argmax(eeg_signal[i:i+40]) + i  # Find the index relative to the full signal
            
            # Set the next 40 points in 'outliers' to 0 (reset outlier flag)
            for j in range(40):
                if i + j < len(outlier_scores):  # Ensure we don't go out of bounds
                    outlier_scores[i+j] = 0
        outlier_scores[temp_outlier] =-1"""
################### REDUCE SERIAL DETECTIONS INTO ONE #####################

###########################################################################
print(outlier_scores)
spike_indices = np.where(outlier_scores == -1)[0]
print(spike_indices)

# Define the confidence threshold
confidence_threshold = 1

# Group consecutive indices
groups = []
current_group = [spike_indices[0]]

for i in range(1, len(spike_indices)):
    if spike_indices[i] <= spike_indices[i - 1] + 20:
        current_group.append(spike_indices[i])
    else:
        groups.append(current_group)
        current_group = [spike_indices[i]]
groups.append(current_group)

# Filter groups based on confidence and get middle indices
filtered_middle_indices = []

for group in groups:
    confidence = len(group)
    if confidence >= confidence_threshold:
        middle_index = group[len(group) // 2]
        filtered_middle_indices.append(middle_index)

print(filtered_middle_indices)

spike_indices = filtered_middle_indices
##### APPLY FOURIER TRANSFORM #####

# Plot signal and detected spikes
def plot(mat_file,result_array,lower_bound,upper_bound):
    colors = ['r', 'm', 'y']
    plt.plot(result_array)
    spike_times = mat_file['spike_times'][0][0][0]
    spike_class = mat_file['spike_class'][0][0][0]
    for spike_idx in spike_indices:
        plt.axvline(x=spike_idx, color='red', linestyle='--')
    for spike, class_label in zip(spike_times, spike_class):
        if lower_bound< spike <= upper_bound:
            plt.axvline(x=spike-lower_bound, color=colors[class_label])

    plt.xlabel('Sample Index')  # Label for the x-axis
    plt.ylabel('Value')   
    plt.show()

plot(mat_file, eeg_signal,low_bound,up_bound)

#print(np.shape(outliers))