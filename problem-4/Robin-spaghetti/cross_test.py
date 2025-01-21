import numpy as np
import matplotlib.pyplot as plt

# Function to generate torus point cloud
def generate_random_torus(R, r, n_points):
    """
    Generate random points within a 3D torus.
    R: Major radius
    r: Minor radius
    n_points: Number of random points to generate
    """
    u = np.random.uniform(0, 2 * np.pi, n_points)
    v = np.random.uniform(0, 2 * np.pi, n_points)
    w = np.random.uniform(-r, r, n_points)  # Random points within the tube radius

    x = (R + w * np.cos(v)) * np.cos(u)
    y = (R + w * np.cos(v)) * np.sin(u)
    z = w * np.sin(v)

    return x, y, z

# Generate torus data
R = 5  # Major radius
r = 2  # Minor radius
n = 10000  # Resolution
x, y, z = generate_random_torus(R, r, n)

# Flatten the arrays
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
print(f"Number of points: {len(x_flat)}")
# Function to get the 2D imprint of points on a rotating plane
def get_rotated_plane_imprints_all_points(x, y, z, n_rotations=100):
    """
    Ensure all points are included by mapping each point to its closest rotated plane.
    n_rotations: Number of rotations (angle increments for the plane)
    """
    angles = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Calculate angles of each point
    point_angles = np.arctan2(y, x)
    point_angles[point_angles < 0] += 2 * np.pi  # Map to [0, 2π]

    # Find the closest rotation for each point
    closest_rotation_indices = np.floor((point_angles / (2 * np.pi)) * n_rotations).astype(int)

    # Create arrays to store 2D imprints
    all_x_proj = []
    all_z_proj = []

    # Process points for each rotation
    for i in range(n_rotations):
        mask = closest_rotation_indices == i

        # Apply the rotation matrix
        x_proj = cos_angles[i] * x[mask] + sin_angles[i] * y[mask]
        z_proj = z[mask]  # Keep Z as is since we project onto the XZ plane

        # Filter for one side of the torus (e.g., positive X-axis side only)
        mask_one_side = x_proj > 0
        x_proj = x_proj[mask_one_side]
        z_proj = z_proj[mask_one_side]

        # Append the projected coordinates
        all_x_proj.append(x_proj)
        all_z_proj.append(z_proj)

    # Combine all projected points
    all_x_proj = np.concatenate(all_x_proj)
    all_z_proj = np.concatenate(all_z_proj)

    return all_x_proj, all_z_proj

# Get the 2D imprints on the rotating plane
imprint_x, imprint_z = get_rotated_plane_imprints_all_points(x_flat, y_flat, z_flat)
# Center the imprints at the origin
centroid_x = np.mean(imprint_x)
centroid_z = np.mean(imprint_z)
imprint_x -= centroid_x
imprint_z -= centroid_z
print(f"Number of imprints: {len(imprint_x)}")
# Plot the 3D torus and its 2D imprints
fig = plt.figure(figsize=(12, 6))

# 3D plot of the torus
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_flat, y_flat, z_flat, s=1, color='blue', alpha=0.5)
ax1.set_title("3D Torus Point Cloud")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Set the ticks for X, Y, and Z axes to be the same
ticks = np.linspace(-R-r, R+r, num=5)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_zticks(ticks)

# 2D plot of the imprints
ax2 = fig.add_subplot(122)
ax2.scatter(imprint_x, imprint_z, s=1, color='red', alpha=0.5)
ax2.set_title("2D Imprints on Rotating Plane (One Side, All Points)")
ax2.set_xlabel("Rotated X'")
ax2.set_ylabel("Rotated Z'")
ax2.axis('equal')

plt.tight_layout()
plt.show()
