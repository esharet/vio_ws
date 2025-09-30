import numpy as np
import matplotlib.pyplot as plt

# --- Rotation Matrices ---
def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])

# --- Plot Function ---
def plot_axes(R, ax, title):
    origin = np.zeros(3)
    x_axis = R @ np.array([1, 0, 0])
    y_axis = R @ np.array([0, 1, 0])
    z_axis = R @ np.array([0, 0, 1])

    ax.quiver(*origin, *x_axis, color="r")
    ax.quiver(*origin, *y_axis, color="g")
    ax.quiver(*origin, *z_axis, color="b")

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title(title)

# --- Angles ---
roll = np.deg2rad(90)
pitch = np.deg2rad(45)
yaw = np.deg2rad(180)

# --- Extrinsic sequence ---
R0 = np.eye(3)
R1 = Rx(roll) @ R0      # roll about world X
R2 = Ry(pitch) @ R1     # then pitch about world Y
R3 = Rz(yaw) @ R2       # then yaw about world Z

R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  # combined
RT = R @ R0
# --- Plot ---
fig = plt.figure(figsize=(10,10))
plot_axes(R0, fig.add_subplot(221, projection="3d"), "Step 1: Rest")
plot_axes(R1, fig.add_subplot(222, projection="3d"), "Step 2: Roll 90° (world X)")
plot_axes(RT, fig.add_subplot(223, projection="3d"), "Step 3: + Pitch 45° (world Y)")
plot_axes(R3, fig.add_subplot(224, projection="3d"), "Step 4: + Yaw 180° (world Z)")
plt.tight_layout()
plt.show()