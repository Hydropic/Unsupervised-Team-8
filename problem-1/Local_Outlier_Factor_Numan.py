import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
import time
__path__ = 'Datasets/sample_1' 
def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)

    return mat

def flatten(arr,variable):
    data = arr[variable]
    data= data.flatten()
    return data


mat_file = load_mat_file(__path__)
eeg_signal_total = flatten(mat_file,'data')
eeg_signal_total = (eeg_signal_total)
#eeg_signal = eeg_signal_total[low_bound:up_bound]
#eeg_signal = (eeg_signal)
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
        temp_array.append(Check_Spikes(dataset[i+interval_start], 3.0))
    
        if i%update_interval == 0:
            print(f"Creating Spike Array : {int(i/update_interval)} of {int(len(dataset[interval_start:interval_length+interval_start])/update_interval)} in {time.time()-start_time} seconds")
            start_time = time.time() # Measure the Performance

    data_array = np.array(temp_array).reshape(1, -1)
    data_array = data_array.flatten()
    return data_array

###############################################################




# Assuming `eeg_signal` is your EEG data (1D array)
window_size = 200  # Define window size
stride = 50  # Define the stride for overlapping windows

# Normalize the data
scaler = StandardScaler()


# Create windows from the EEG signal
def create_windows(data, window_size, stride):
    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

from sklearn.neighbors import LocalOutlierFactor
#Reshape data for LOF


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


# data_points = eeg_signal_total.reshape(-1, 1)

# lof = LocalOutlierFactor(n_neighbors=45)
# outlier_scores = lof.fit_predict(data_points)

# ################### REDUCE SERIAL DETECTIONS INTO ONE #####################
# for i in range(len(outlier_scores)):
#     if outlier_scores[i] == -1:  # Check if current point is an outlier
#         # Find the index of the max value in the next 40 points of the signal
#         if i + 40 < len(eeg_signal):
#             temp_outlier = np.argmax(eeg_signal[i:i+40]) + i  # Find the index relative to the full signal
            
#             # Set the next 40 points in 'outliers' to 0 (reset outlier flag)
#             for j in range(40):
#                 if i + j < len(outlier_scores):  # Ensure we don't go out of bounds
#                     outlier_scores[i+j] = 0
#         outlier_scores[temp_outlier] =-1
# ###########################################################################



# Iterate through the outliers array
#spike_indices = np.where(outlier_scores == -1)[0]
##### APPLY FOURIER TRANSFORM #####


def Merge_detections(data ,detect_start, detect_stop, window_size):
    j=1
    #spike_indices_merged = []
    outlier_scores_merged = np.array([])
    step_count = (detect_stop-detect_start)//window_size
    while j <= step_count:
        start_time = time.time()
        temp_data = data[detect_start+(j-1)*window_size:detect_start+(j)*window_size]
        data_points = temp_data.reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=301)  # standard value for window_size 10000 is 45
        temp_outlier_scores = lof.fit_predict(data_points)
        for i in range(len(temp_outlier_scores)):
            if temp_outlier_scores[i] == -1:  # Check if current point is an outlier
        # Find the index of the max value in the next 40 points of the signal
                if i + 40 < len(temp_data):
                    temp_outlier = np.argmax(temp_data[i:i+40]) + i  # Find the index relative to the full signal
                    
                    # Set the next 40 points in 'outliers' to 0 (reset outlier flag)
                    for k in range(40):
                        if i + k < len(temp_outlier_scores):  # Ensure we don't go out of bounds
                            temp_outlier_scores[i+k] = 0
                temp_outlier_scores[temp_outlier] =-1
        outlier_scores_merged = np.concatenate((outlier_scores_merged,temp_outlier_scores))
        print(f"Calculating Local Outlier Factors: {j} of {step_count} in {time.time()-start_time} seconds")
        start_time= time.time()
        j+=1 
    return outlier_scores_merged

low_bound = 0
up_bound = len(eeg_signal_total)
window_size = 1500#10000    ### 848 windowsize retrieved by maximizing tp (n_neighbours = 140) and z_score threshold of 3.0###
eeg_signal = Create_spikeArray(eeg_signal_total,low_bound,up_bound-low_bound,1000)
eeg_signal = scaler.fit_transform(eeg_signal.reshape(-1, 1)).flatten()
data_points = eeg_signal.reshape(-1, 1)

outlier_scores_merged = Merge_detections(eeg_signal, low_bound, up_bound,window_size)
spike_indices = np.where(outlier_scores_merged==-1)[0]


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

#plot(mat_file, eeg_signal,low_bound,up_bound)





def count_fn_fp(ground_truth, predictions, tolerance=40):
    """
    Count the number of false positives, false negatives, and true positives.
    
    Parameters:
    - ground_truth (numpy array): Ground truth spike times.
    - predictions (numpy array): Predicted spike times.
    - tolerance (int): Tolerance window in points for matching a prediction to ground truth.
    
    Returns:
    - false_negative (int): Number of false negatives.
    - false_positive (int): Number of false positives.
    - true_positive (int): Number of true positives.
    """
    
    matched_spikes = set()
    true_positive_spikes = set()
    
    update_interval = 1000
    start_time = time.time()

    # Loop over each ground truth spike
    for idx, d in enumerate(ground_truth):
        found_match = False
        for p in predictions:
            if d - tolerance/5 <= p <= d + 4*tolerance/5:
                # True positive found
                matched_spikes.add(d)
                true_positive_spikes.add(p)
                found_match = True
                break  # Move to the next ground truth spike once a match is found

        # Log progress
        if idx % update_interval == 0:
            print(f"Checking accuracy: {idx // update_interval} of {len(ground_truth) // update_interval} in {time.time() - start_time:.2f} seconds")
            start_time = time.time()

    # Calculate counts
    false_negative = len(ground_truth) - len(matched_spikes)
    false_positive = len(predictions) - len(true_positive_spikes)
    true_positive = len(true_positive_spikes)
    
    print(f"False Negatives: {false_negative}, False Positives: {false_positive}, True Positives: {true_positive}")
    
    return false_negative, false_positive, true_positive


def fast_count_fp_fn_tp(ground_truth, predictions, tolerance=40):
    predictions = np.sort(predictions)  # Ensure sorted for easier traversal
    ground_truth = np.sort(ground_truth)
    
    tp, fp, fn = 0, 0, 0
    i, j = 0, 0

    while i < len(predictions) and j < len(ground_truth):
        if abs(predictions[i] - ground_truth[j]) <= tolerance:
            # True Positive: Match found within the window range
            tp += 1
            i += 1
            j += 1
        elif predictions[i] < ground_truth[j]:
            # False Positive: Prediction with no matching ground truth within window
            fp += 1
            i += 1
        else:
            # False Negative: Ground truth with no matching prediction within window
            fn += 1
            j += 1

    # Remaining unmatched predictions and ground truths
    fp += len(predictions) - i
    fn += len(ground_truth) - j

    print(f"False Negatives: {fn}, False Positives: {fp}, True Positives: {tp}")
    return fp, fn, tp
   

# Example usage
fn, fp, tp = fast_count_fp_fn_tp(mat_file['spike_times'][0][0][0], spike_indices, 40)

threshold_param = 2.7
left_interval = 30
right_interval = 20

# data
data = mat_file['data'][0]
spike_times = mat_file['spike_times'][0][0][0]


# mean and stdv
mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0)


detected_spikes = []

for t in spike_indices:
    spike = data[t-left_interval:t+right_interval]
    detected_spikes.append(spike)

#detected_spikes = np.array(detected_spikes)

covariance = np.cov(detected_spikes, rowvar=False)
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.imshow(covariance)
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(covariance)
eigenvalues = np.sort(eigenvalues)[::-1]
print(eigenvalues)

plt.plot(np.linspace(1, len(eigenvalues), num=len(eigenvalues)), eigenvalues)

plt.show()
#print(np.shape(outliers))
