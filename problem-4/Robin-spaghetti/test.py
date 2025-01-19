import numpy as np
import matplotlib.pyplot as plt

# Parameters for the torus
R = 5  # Major radius (distance from the center of the tube to the center of the torus)
r = 2  # Minor radius (radius of the tube)

# Number of points to generate in each cross-section
num_points = 100
# Number of layers (cross-sections) along the z-axis
num_layers = 100
# Range of z values (height of the torus)
z_values = np.linspace(-r, r, num_layers)

# Initialize arrays for point cloud data
x_points = []
y_points = []
z_points = []

# Generate the point cloud by iterating through z values and corresponding theta values
for z in z_values:
    # Calculate the radius at this z-height (due to the shape of the torus)
    radius = R + r * np.cos(np.arcsin(z / r))
    
    # Generate points along the circle for this z-slice
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Append the points to the arrays
    x_points.extend(x)
    y_points.extend(y)
    z_points.extend([z] * num_points)

# Convert lists to numpy arrays for plotting
x_points = np.array(x_points)
y_points = np.array(y_points)
z_points = np.array(z_points)

# Plot the point cloud in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points in the 3D space
ax.scatter(x_points, y_points, z_points, c=z_points, cmap='viridis', s=1)
ax.set_title('Torus Point Cloud')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
