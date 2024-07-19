import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style

# Path to the directory containing the CSV files
data_directory = '../TrainData/hand_data/'

# Filename
file_name = 'hand_gesture_data_Fri_Jun_14_120815_2024.csv'

style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define, which landmarks form which finger
thumb = [0, 1, 2, 3, 4]
index = [5, 6, 7, 8]
middle = [9, 10, 11, 12]
ring = [13, 14, 15, 16]
pinky = [17, 18, 19, 20]
palm = [0, 5, 9, 13, 17, 0]

with open(data_directory + file_name, 'r') as file:
    reader = csv.reader(file)
    # convert to np array and generate random index
    data = np.array(list(reader))
    rand_idx = np.random.randint(1, len(data))
    print(f'sample no {rand_idx}')

    # read label and landmarks from random sample
    label = data[rand_idx][0]
    rand_sample =  data[rand_idx][1:]

    # iterate the landmarks and convert them to float values
    pose = []
    for landmark in rand_sample:
        xs, ys, zs = [float(x) for x in landmark.split(',')]
        ax.scatter(xs, ys, zs, color='black')
        pose.append([xs,ys,zs])

    """thumb = [[pose[0][0], pose[1][0], pose[2][0], pose[3][0], pose[4][0]],
             [pose[0][1], pose[1][1], pose[2][1], pose[3][1], pose[4][1]],
             [pose[0][2], pose[1][2], pose[2][2], pose[3][2], pose[4][2]]]"""
    for finger in [thumb, index, middle, ring, pinky, palm]:
        line_x = []
        line_y = []
        line_z = []

        for joint in finger:
            line_x.append(pose[joint][0])
            line_y.append(pose[joint][1])
            line_z.append(pose[joint][2])

        ax.plot(line_x, line_y, line_z, color='grey')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(-1,0)
plt.title(label=label)
plt.show()

