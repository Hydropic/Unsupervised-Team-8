import scipy.io
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd # To check the Keys etc.

sample_1 = scipy.io.loadmat('Datasets/sample_1.mat')

################Check Keys and General Shape of the Data######################

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


##############################################################################

############## Plot the variable 'data' (1,2880000) ##########################

data = sample_1['data']
data_flat = data.flatten()

# Plot the flattened data
plt.plot(data_flat)
plt.xlabel('Sample Index')  # Label for the x-axis
plt.ylabel('Value')         # Label for the y-axis
plt.title('Plot of Data with Shape (1, 2880000)')
plt.show()

##############################################################################

