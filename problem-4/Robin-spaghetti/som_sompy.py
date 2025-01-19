import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu

# Parameters for the torus
R = 10  # Major radius
r = 2   # Minor radius

# Generate torus point cloud
u = np.linspace(0, 2 * np.pi, 50)  # Orientation
v = np.linspace(0, 2 * np.pi, 50)  # Cross-section
u, v = np.meshgrid(u, v)

X = (R + r * np.cos(v)) * np.cos(u)
Y = (R + r * np.cos(v)) * np.sin(u)
Z = r * np.sin(v)

# Flatten the arrays for point cloud representation
points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# Normalize the data
points_norm = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

# Initialize and train the SOM with a toroidal topology
som_size = 20  # Size of the SOM grid
som = somoclu.Somoclu(som_size, som_size, compactsupport=False, gridtype='toroid')
som.train(points_norm)

# Get the SOM weights
som_weights = som.codebook.reshape(som_size, som_size, -1)

# Denormalize the SOM weights to match the original data scale
som_weights_denorm = som_weights * (points.max(axis=0) - points.min(axis=0)) + points.min(axis=0)

# Plot the torus point cloud and SOM nodes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='b', marker='o', s=1, label='Torus Point Cloud')

# Plot the SOM nodes
ax.scatter(som_weights_denorm[:, :, 0].flatten(), som_weights_denorm[:, :, 1].flatten(), som_weights_denorm[:, :, 2].flatten(), c='r', marker='x', s=50, label='SOM Nodes')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Torus Point Cloud with SOM Mapping')
ax.legend()

plt.show()