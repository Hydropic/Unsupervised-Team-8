import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_mat_file(file_path):
    mat = loadmat(file_path)
    return mat

def calculate_mean(arr):
    return np.mean(arr, axis=0)

def calculate_std_dev(arr):
    return np.std(arr, axis=0)

def calculate_z_score(points, dataset):
    mean = calculate_mean(dataset)
    std_dev = calculate_std_dev(dataset)
    return (points - mean) / std_dev

def plot_spike_regions(file_path, variable_name, interval_length=100, threshold_factor=2):
    mat_file = load_mat_file(file_path)
    data = mat_file[variable_name][0]
    plt.plot(data[:1000])

    mean = calculate_mean(data)
    std_dev = calculate_std_dev(data)
    threshold = mean + threshold_factor * std_dev

    num_intervals = len(data) // interval_length
    """for i in range(num_intervals):
        interval = data[i*interval_length:(i+1)*interval_length]
        z_scores = calculate_z_score(interval, data)
        
        spikes = np.abs(z_scores) > threshold_factor*threshold
        #plt.plot(interval[spikes], 'ro')
        #plt.plot(interval, 'b-')"""
    plt.show()


def plot_ground_truth(file_path):
    pass

# plot_spike_regions('Unsupervised-Team-8/sample_1.mat', 'data', interval_length=100, threshold_factor=2)