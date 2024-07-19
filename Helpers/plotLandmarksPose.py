import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

# Filenames for left and right samples
path_left = '../TrainData/test_left/dir_data_left_2024_Jul_19_103951.csv'
path_right = '../TrainData/test_right/dir_data_right_2024_Jul_19_103951.csv'

# style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sample_idx = 5

# Define the pose landmarks
pose_landmarks = [
    (0, 'nose'), (1, 'left_eye_inner'), (2, 'left_eye'), (3, 'left_eye_outer'),
    (4, 'right_eye_inner'), (5, 'right_eye'), (6, 'right_eye_outer'), (7, 'left_ear'),
    (8, 'right_ear'), (9, 'mouth_left'), (10, 'mouth_right'), (11, 'left_shoulder'),
    (12, 'right_shoulder'), (13, 'left_elbow'), (14, 'right_elbow'), (15, 'left_wrist'),
    (16, 'right_wrist'), (17, 'left_pinky'), (18, 'right_pinky'), (19, 'left_index'),
    (20, 'right_index'), (21, 'left_thumb'), (22, 'right_thumb'), (23, 'left_hip'),
    (24, 'right_hip'), (25, 'left_knee'), (26, 'right_knee'), (27, 'left_ankle'),
    (28, 'right_ankle'), (29, 'left_heel'), (30, 'right_heel'), (31, 'left_foot_index'),
    (32, 'right_foot_index')
]

def read_sample(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader))
        rand_idx = np.random.randint(1, len(data))
        print(f'Sample no {sample_idx} from {file_name}')

        # Read label and landmarks from random sample
        label = data[sample_idx][:1]
        rand_sample = data[sample_idx][2:]

        # Iterate the landmarks and convert them to float values
        pose = []
        for landmark in rand_sample:
            xs, ys, zs = [float(x) for x in landmark.split(',')]
            pose.append([xs, ys, zs])
        return label, pose

label_left, pose_left = read_sample(path_left)
label_right, pose_right = read_sample(path_right)

# Plotting the left pose
for idx, (x, y, z) in enumerate(pose_left):
    ax.scatter(x, y, z, color='blue', label='Left' if idx == 0 else "")

# Plotting the right pose
for idx, (x, y, z) in enumerate(pose_right):
    ax.scatter(x, y, z, color='red', label='Right' if idx == 0 else "")

# Connect the landmarks with lines
for start_idx, _ in pose_landmarks:
    if start_idx >= len(pose_left) or start_idx >= len(pose_right):
        continue

    # Left sample lines
    ax.plot([pose_left[start_idx][0]], [pose_left[start_idx][1]], [pose_left[start_idx][2]], marker='o', color='blue')

    # Right sample lines
    ax.plot([pose_right[start_idx][0]], [pose_right[start_idx][1]], [pose_right[start_idx][2]], marker='o', color='red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-1, 0)
plt.legend()
plt.title(f'Left (Blue) vs Right (Red)\nLeft Label: {label_left}, Right Label: {label_right}')

plt.show()
