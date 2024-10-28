from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import make_scorer

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

def Create_spikeArray(dataset, interval_start, interval_length, update_interval, z_score):
    temp_array = []
    start_time = time.time() #Measure the Performance
    for i in range(len(dataset[interval_start:interval_length+interval_start])):         
        temp_array.append(Check_Spikes(dataset[i+interval_start], z_score))
    
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



import optuna
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
eeg_signal_total_scaled = scaler.fit_transform(eeg_signal_total.reshape(-1, 1))




def Merge_detections(data ,detect_start, detect_stop, window_size, n_neighbors):
    j=1
    #spike_indices_merged = []
    outlier_scores_merged = np.array([])
    step_count = (detect_stop-detect_start)//window_size
    while j <= step_count:
        start_time = time.time()
        temp_data = data[detect_start+(j-1)*window_size:detect_start+(j)*window_size]
        data_points = temp_data.reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors)
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
#window_size =10000
#eeg_signal = Create_spikeArray(eeg_signal_total,low_bound,up_bound-low_bound,1000)
#eeg_signal = scaler.fit_transform(eeg_signal.reshape(-1, 1)).flatten()
#data_points = eeg_signal.reshape(-1, 1)



def count_fn_fp(ground_truth, predictions, tolerance=40):

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
    
    #print(f"False Negatives: {false_negative}, False Positives: {false_positive}, True Positives: {true_positive}")
    
    return false_negative, false_positive, true_positive

# Example usage

default_window_size = 10000
default_n_neighbors = 20
default_z_score_threshold = 3.0

def Create_spikeArray(dataset, interval_start, interval_length, update_interval, z_score):
    temp_array = []
    start_time = time.time() #Measure the Performance
    for i in range(len(dataset[interval_start:interval_length+interval_start])):         
        temp_array.append(Check_Spikes(dataset[i+interval_start], z_score))
    
        if i%update_interval == 0:
            print(f"Creating Spike Array : {int(i/update_interval)} of {int(len(dataset[interval_start:interval_length+interval_start])/update_interval)} in {time.time()-start_time} seconds")
            start_time = time.time() # Measure the Performance

    data_array = np.array(temp_array).reshape(1, -1)
    data_array = data_array.flatten()
    return data_array

low_bound = 0
up_bound = len(eeg_signal_total)
#window_size = 848#10000

def objective(trial):
    # Attempt to suggest values for hyperparameters, fallback to defaults
    window_size = trial.suggest_int('window_size', 10, 100000) if trial else default_window_size
    n_neighbors = trial.suggest_int('n_neighbors', 5, 200) if trial else default_n_neighbors
    z_score_threshold = trial.suggest_float('z_scores', 0.0, 5.0) if trial else default_z_score_threshold
    eeg_signal = Create_spikeArray(eeg_signal_total,low_bound,up_bound-low_bound,1000,z_score= z_score_threshold)
    
    outlier_scores_merged = Merge_detections(eeg_signal, low_bound, up_bound,window_size = window_size, n_neighbors=n_neighbors)
    #lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    spike_indices = np.where(outlier_scores_merged==-1)[0]
    # Fit and predict
    #outliers = lof.fit_predict(eeg_signal_total_scaled)  # Outliers are marked as -1
    
    # Custom scoring: Ratio of outliers detected (adjust as necessary for dataset)
    #outlier_score = np.mean(outliers == -1)  # For example, mean number of outliers detected
    
    fn, fp, tp = count_fn_fp(mat_file['spike_times'][0][0][0], spike_indices, tolerance=50)
    pre = tp/(fp+tp)
    rec = tp/(tp+fn)
    return (2*pre*rec/(pre+rec)) # Return a value to maximize or minimize



study = optuna.create_study(direction='maximize')  # Maximize outlier detection
study.optimize(objective, n_trials=10)  # Run for 100 trials

print("Best Parameters:", study.best_params)
print("Best Outlier Detection Score:", study.best_value)





