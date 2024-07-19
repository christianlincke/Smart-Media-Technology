"""
Plot two pose files to compare their data.
Useful toi check if mirroring works as intended

Last changes by Christian 19. Jul 24
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from Models import ArmModel3D
from mpl_toolkits.mplot3d import Axes3D


# Plotting mode - 3d or 2d
DIMENSIONS = '2d'

# Random Sample?
RND = True

# if not random, Which sample shall be evaluated?
SAMPLE_IDX = 5

# Filenames for left and right samples
path_left = '../TrainData/test_left/dir_data_left_2024_Jul_19_120402.csv'
path_right = '../TrainData/test_right/dir_data_right_2024_Jul_19_120402.csv'

# Get landmark masks
mask_left = ArmModel3D.landmarkMask('left')
mask_right = ArmModel3D.landmarkMask('right')

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
        label = data[sample_idx][:2]
        rand_sample = data[sample_idx][2:]

        # Iterate the landmarks and convert them to float values
        pose = []
        for landmark in rand_sample:
            xs, ys, zs = [float(point) for point in landmark.split(',')]
            pose.append([xs, ys, zs])
        return label, np.array(pose), sample_idx


label_left, pose_left, new_idx = read_sample(path_left, SAMPLE_IDX, True)
label_right, pose_right, _ = read_sample(path_right, new_idx, False)

# Extract landmarks using the masks
print("Pre mask: ", len(pose_left), len(pose_right))
pose_left = pose_left[mask_left]
pose_right = pose_right[mask_right]
print("Post mask: ", len(pose_left), len(pose_right))

# Plotting the left pose
for idx, (x, y, z) in enumerate(pose_left):
    if DIMENSIONS.lower() == '3d':
        ax.scatter(x, y, z, color='blue', label='Left' if idx == 0 else "")
    else:
        ax.scatter(x, y, color='blue', label='Left' if idx == 0 else "")

# Plotting the right pose
for idx, (x, y, z) in enumerate(pose_right):
    if DIMENSIONS.lower() == '3d':
        ax.scatter(x, y, z, color='red', label='Right' if idx == 0 else "")
    else:
        ax.scatter(x, y, color='red', label='Right' if idx == 0 else "")


ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

if DIMENSIONS.lower() == '3d':
    ax.set_zlabel('z')
    ax.set_zlim(-1, 0)

plt.legend()
plt.title(f'Left (Blue) vs Right (Red)\nLeft Label: {label_left}, Right Label: {label_right}')

plt.show()
