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

# Function to get the 2D imprint of points on a rotating plane
def get_rotated_plane_imprints(x, y, z, n_rotations=100, plane_tolerance=0.1):
    """
    Rotate a plane around the Z-axis and collect the 2D imprints of the points.
    n_rotations: Number of rotations (angle increments for the plane)
    plane_tolerance: Tolerance to consider points near the plane
    """
    angles = np.linspace(0, 2 * np.pi, n_rotations)
    all_imprint_coords = []

    for angle in angles:
        # Rotation matrix for the Z-axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        # Rotate the points to align with the current plane
        x_rot = cos_angle * x + sin_angle * y
        y_rot = -sin_angle * x + cos_angle * y

        # Select points near the rotated plane (projecting onto XZ)
        mask = np.abs(y_rot) < plane_tolerance

        # Filter for one side of the torus (e.g., positive X-axis side only)
        mask = mask & (x_rot > 0)  # Keep only one side of the torus

        x_proj = x_rot[mask]
        z_proj = z[mask]  # Keep Z as is since the plane is still in XZ

        # Transform back into the rotating plane's 2D coordinate space
        xy_proj = np.vstack((x_proj, z_proj)).T
        all_imprint_coords.append(xy_proj)

    # Combine all 2D imprints
    all_imprint_coords = np.vstack(all_imprint_coords)
    return all_imprint_coords[:, 0], all_imprint_coords[:, 1]  # Return as (X', Z')

# Get the 2D imprints on the rotating plane
imprint_x, imprint_z = get_rotated_plane_imprints(x_flat, y_flat, z_flat)

# Plot the 3D torus and its 2D imprints
fig = plt.figure(figsize=(12, 6))

# 3D plot of the torus
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_flat, y_flat, z_flat, s=1, color='blue', alpha=0.5)
ax1.set_title("3D Torus Point Cloud")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# 2D plot of the imprints
ax2 = fig.add_subplot(122)
ax2.scatter(imprint_x, imprint_z, s=1, color='red', alpha=0.5)
ax2.set_title("2D Imprints on Rotating Plane (One Side)")
ax2.set_xlabel("Rotated X'")
ax2.set_ylabel("Rotated Z'")
ax2.axis('equal')

plt.tight_layout()
plt.show()
