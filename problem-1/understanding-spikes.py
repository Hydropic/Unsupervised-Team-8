from utils import *


def groundtruth(lower_bound=0, upper_bound=1000):
    # ploting raw data
    # plot_spike_regions('sample_1.mat', 'data')

    # plotting annotated data

    colors = ['r', 'm', 'y']
    mat_file = load_mat_file('../sample_1.mat')
    data = mat_file['data'][0]

    plt.plot(data[lower_bound:upper_bound])

    spike_times = mat_file['spike_times'][0][0][0]
    spike_class = mat_file['spike_class'][0][0][0]

    for spike, class_label in zip(spike_times, spike_class):
        if lower_bound < spike < upper_bound:
            plt.axvline(x=spike-lower_bound, color=colors[class_label])

    plt.show()

    print("stop")


def main():
    groundtruth(lower_bound=500, upper_bound=600)


if __name__ == '__main__':
    main()
