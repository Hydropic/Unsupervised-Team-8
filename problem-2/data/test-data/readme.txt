Synthetic maternal abdomen multi electrode ECG recordings

* Each file is an independent recording of a patient.
* Each file contains 4 electrodes and therefore 4 recorded channels.
* The sampling rate of each channel is 360 Hz.
* Some of the fetal heartbeat signals contain pathologies, try to detect them.

FYI: The synthetic data you got was scaled in magnitude during preprocessing, so this is only for those of you who are interested in the details. Since the recorded data comes from an analogue real world signal, it had to be digitized. The original signal has been recorded in millivolts [mV], the recording range was roughly -5mv to 5mv. The data has been digitized with 11 bit precision, which leaves 2^11 = 2048 different buckets to put each input reading in. This means
    * a value of -5 mV was translated to 0 and a value of 5 mv was translated to 2047
    * each 1 mV interval from the recorded data is represented with a resolution of 2048/10 = roughly 204 units
    * the unit of the data you got is therefore [ADU/mv], which means analogue to digital units per millivolt

