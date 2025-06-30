"""
Plot two pose files to compare their data.
Useful toi check if mirroring works as intended

Last changes by Christian 19. Jul 24
"""
import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from Models import ArmModel
from mpl_toolkits.mplot3d import Axes3D
mpl.use('Qt5Agg')

# What parameter?
PARAM = "dir" # or stretch

# Plotting mode - 3d or 2d
DIMENSIONS = '3d'

# Random Sample?
RND = True

if PARAM == "dir":
    num_labels = 2
else:
    num_labels = 1

# if not random, Which sample shall be evaluated?
SAMPLE_IDX = 5

# Filenames for left and right samples
path_left = '../TrainData/dir_data_left/dir_data_left_2024_Jul_23_115005.csv'
path_right = '../TrainData/dir_data_right/dir_data_right_2024_Jul_23_115005.csv'

# Get landmark masks
mask_left = np.array([11, 13]) # ArmModel.landmarkMask('left')
mask_right = np.array([12, 14]) # ArmModel.landmarkMask('right')

# Init matplotlib stuff
fig = plt.figure()
if DIMENSIONS.lower() == '3d':
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)

def read_sample(file_name, sample_idx, generate_new_rnd):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader))

        if RND and generate_new_rnd:
            sample_idx = np.random.randint(len(data))

        print(f'Sample no {sample_idx} from {file_name}')

        # Read label and landmarks from random sample
        label = data[sample_idx][:num_labels]
        rand_sample = data[sample_idx][num_labels:]

        # Iterate the landmarks and convert them to float values
        pose = []
        for landmark in rand_sample:
            xs, ys, zs = [float(point) for point in landmark.split(',')]
            pose.append([xs, ys, zs])
        return label, np.array(pose), sample_idx

def azimuth_elevation(p1, p2):
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    vz = p2[2] - p1[2]

    # Adjust coordinate system: facing toward -Z
    azimuth = math.atan2(vx, -vz)  # 0° = forward, +90° = right
    horizontal_dist = math.hypot(vx, vz)
    elevation = -math.atan2(vy, horizontal_dist) # vertical image coordinates are inverted, so we need a (-) prefix

    # Convert to degrees
    azimuth_deg = math.degrees(azimuth)
    elevation_deg = math.degrees(elevation)

    # Normalize azimuth to [-180°, +180°]
    if azimuth_deg > 180:
        azimuth_deg -= 360

    return azimuth_deg, elevation_deg

label_left, pose_left, new_idx = read_sample(path_left, SAMPLE_IDX, True)
label_right, pose_right, _ = read_sample(path_right, new_idx, False)

# calculate angles
az_left, el_left = azimuth_elevation(pose_left[11], pose_left[13])

# Extract landmarks using the masks
print("Pre mask: ", len(pose_left), len(pose_right))
pose_left = pose_left[mask_left]
pose_right = pose_right[mask_right]
print("Post mask: ", len(pose_left), len(pose_right))
print(pose_left)
print(pose_right)

###TESTING###
# pose_left = np.array([[0, 0, 0],[1, 1, 1]])

# Plotting the left pose
for idx, (x, y, z) in enumerate(pose_left):
    if DIMENSIONS.lower() == '3d':
        ax.scatter(x, y, z, color='blue', label='Left' if idx == 0 else "")
        ax.text(x, y, z, idx, size=10, zorder=1, color='k')
    else:
        ax.scatter(x, y, color='blue', label='Left' if idx == 0 else "")
        ax.annotate(mask_left[idx], (x, y))
ax.plot(pose_left[:, 0], pose_left[:, 1], pose_left[:, 2], color='red')

# Plotting the right pose
for idx, (x, y, z) in enumerate(pose_right):
    if DIMENSIONS.lower() == '3d':
        ax.scatter(x, y, z, color='red', label='Right' if idx == 0 else "")
    else:
        ax.scatter(x, y, color='red', label='Right' if idx == 0 else "")
        ax.annotate(mask_right[idx], (x, y))

print(f"Azimuth: {az_left} Elevation {el_left}")

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_xlim(0, 1)
ax.set_ylim(1, 0)

ax.view_init(vertical_axis="y")

if DIMENSIONS.lower() == '3d':
    ax.set_zlabel('z')
    ax.set_zlim(-1, 0)

plt.legend()
plt.title(f'Left (Blue) vs Right (Red)\nLeft Label: {label_left}, Right Label: {label_right}')

plt.show()