import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Torus parameters
R = 10  # Major radius
r = 2   # Minor radius

# Function to generate random points on the torus
def generate_torus_points(R, r, num_points):
    u = np.random.uniform(0, 2 * np.pi, num_points)  # Orientation
    v = np.random.uniform(0, 2 * np.pi, num_points)  # Cross-section
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.vstack((x, y, z)).T

# Function to project a point onto the torus surface
def project_to_torus(point, R, r):
    x, y, z = point
    theta = np.arctan2(y, x)  # Compute angle in the XY plane
    d = np.sqrt(x**2 + y**2)  # Distance from the origin in the XY plane
    phi = np.arctan2(z, d - R)  # Angle in the cross-sectional circle
    x_proj = (R + r * np.cos(phi)) * np.cos(theta)
    y_proj = (R + r * np.cos(phi)) * np.sin(theta)
    z_proj = r * np.sin(phi)
    return np.array([x_proj, y_proj, z_proj])
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Function to generate random points on the torus
def generate_torus_points(R, r, num_points):
    u = np.random.uniform(0, 2 * np.pi, num_points)  # Orientation
    v = np.random.uniform(0, 2 * np.pi, num_points)  # Cross-section
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.vstack((x, y, z)).T

# Function to compute the toroidal distance between two points
def toroidal_distance(p1, p2, R, r):
    # Convert 3D Cartesian points to toroidal coordinates
    u1 = np.arctan2(p1[1], p1[0])
    v1 = np.arctan2(p1[2], np.sqrt(p1[0]**2 + p1[1]**2) - R)
    u2 = np.arctan2(p2[1], p2[0])
    v2 = np.arctan2(p2[2], np.sqrt(p2[0]**2 + p2[1]**2) - R)
    
    # Compute angular differences with wrapping
    du = min(abs(u1 - u2), 2 * np.pi - abs(u1 - u2))
    dv = min(abs(v1 - v2), 2 * np.pi - abs(v1 - v2))
    
    # Compute distance in toroidal space
    return np.sqrt((R * du)**2 + (r * dv)**2)

# Generate torus point cloud
points = generate_torus_points(R, r, 5000)

# Normalize the data
points_norm = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

# Initialize SOM with weights on the torus surface
som_size = 20
som = MiniSom(x=som_size, y=som_size, input_len=3, sigma=1.0, learning_rate=0.5)
som_weights = np.zeros((som_size, som_size, 3))

for i in range(som_size):
    for j in range(som_size):
        # Randomly initialize weights on the torus
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, 2 * np.pi)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        som_weights[i, j, :] = [x, y, z]

som.weights = som_weights  # Set SOM weights

# Train SOM and constrain weights to the torus during training
for i in range(1000):  # 1000 training iterations
    rand_idx = np.random.randint(0, len(points_norm))
    som.update(points_norm[rand_idx], som.winner(points_norm[rand_idx]), i, 1000)
    for x in range(som_size):
        for y in range(som_size):
            som.weights[x, y, :] = project_to_torus(som.weights[x, y, :], R, r)

# Plot the torus point cloud and SOM nodes with local connections
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot torus point cloud
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', s=1, label='Torus Point Cloud')

# Plot SOM nodes
som_x = som_weights[:, :, 0]
som_y = som_weights[:, :, 1]
som_z = som_weights[:, :, 2]

ax.scatter(som_x.flatten(), som_y.flatten(), som_z.flatten(), c='r', marker='x', s=50, label='SOM Nodes')

# Visualize local connections between SOM nodes
for i in range(som_size):
    for j in range(som_size):
        # Connect to the right neighbor
        if j < som_size - 1:
            dist = toroidal_distance(som_weights[i, j], som_weights[i, j + 1], R, r)
            if dist < 2 * r:  # Threshold for local connection
                ax.plot(
                    [som_x[i, j], som_x[i, j + 1]],
                    [som_y[i, j], som_y[i, j + 1]],
                    [som_z[i, j], som_z[i, j + 1]],
                    c='gray', linewidth=0.5
                )
        # Connect to the bottom neighbor
        if i < som_size - 1:
            dist = toroidal_distance(som_weights[i, j], som_weights[i + 1, j], R, r)
            if dist < 2 * r:  # Threshold for local connection
                ax.plot(
                    [som_x[i, j], som_x[i + 1, j]],
                    [som_y[i, j], som_y[i + 1, j]],
                    [som_z[i, j], som_z[i + 1, j]],
                    c='gray', linewidth=0.5
                )

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Torus Point Cloud with Local SOM Node Connections')
ax.legend()
plt.show()
