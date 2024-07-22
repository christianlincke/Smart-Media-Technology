"""
Collect train data for the direction detection.

Setting AUG = True will save a mirrored copy of the same dataset.
E.g. recording for 'right' arm will also generate a csv file in
the TrainData/dir_data_left/ folder with the same timestamp.

Other than that, there's a TESTING mode that will record to test_{SIDE} instead.

Last changes by Christian 19. Jul 24
"""
import cv2
import mediapipe as mp
from Models import ArmModel3D
import numpy as np
import time
import csv

# Testing mode - is the code doing the right thing?
# Test data will be save in TrainData/test_*/
TESTING = True ## PLEASE DON'T CHANGE THIS UNLESS YOU'RE SURE HTE CODE IS WORKING

# Which arm should be trained?
ARM = 'right'  # 'left' or 'right'

# Shall augmented data be saved (seperate file)?
# Uses mirrored left data to fake right data
AUG = True

# Global variable to set record time for each gesture in seconds
RECORD_TIME = 2

# Define left/right swap
SWAP = {'right': 'left',
        'left': 'right',}

# Targets to be recorded
# [[az, el], [..]]
target_values = [[-0.5, 0.0],   [-0.25, 0.0],   [0.0, 0.0],     [0.25, 0.0],    [0.5, 0.0],
                 [-0.5, 0.25],  [-0.25, 0.25],  [0.0, 0.25],    [0.25, 0.25],   [0.5, 0.25],
                 [-0.5, -0.25], [-0.25, -0.25], [0.0, -0.25],   [0.25, -0.25],  [0.5, -0.25],
                 [0.0, 0.5],    [0.0, -0.5]
                 ]

target_names = ['90° left / 0°',        '45° left /  0°',       '0° / 0°',          '45° right / 0°',       '90° right / 0°',
                '90° left / 45° up',    '45° left /  45° up',   '0° / 45° up',      '45° right / 45° up',   '90° right / 45° up',
                '90° left / 45° down',  '45° left /  45° down', '0° / 45° down',    '45° right / 45° down', '90° right / 45° down',
                '90° up',               '90° down'
                ]

# Directory to save the data to
if TESTING:
    path = f'TrainData/test_{ARM}/'
    path_mir = f'TrainData/test_{SWAP[ARM]}/'
else:
    path = f'TrainData/dir_data_{ARM}/'
    path_mir = f'TrainData/dir_data_{SWAP[ARM]}/'

# Define how many landmarks we're using
num_landmarks = 33

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

# Function to capture landmarks for a set duration
def record_landmarks(duration, target):
    start_time = time.time()
    landmarks_list = []
    landmarks_list_mir = []

    target_mir = target
    target_mir[0] = -target_mir[0] # mirror coordinate vertically

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose.
        results = pose.process(image)
        results_mir = pose.process(cv2.flip(image, 1))

        # Draw pose landmarks.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark])
            landmarks_mir = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results_mir.pose_landmarks.landmark])
            landmarks_list.append((landmarks, target))
            landmarks_list_mir.append((landmarks_mir, target_mir))

        # flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display the image with instructions.
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording {ARM} arm {target}... {duration - int(elapsed_time)}s left. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Pose Gesture Calibration', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    return landmarks_list, landmarks_list_mir

# Data collection with automated recording
def calibrate_gesture(target, gesture_name, record_time):
    """

    :param target: float pose target value
    :param gesture_name: str pose name
    :param record_time: int recording time
    :return: landmarks list of recorded lanmarks
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display instructions
        cv2.putText(frame, f"Show {ARM} arm {gesture_name} and press 'r' to record. Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Pose Gesture recording', frame)

        if cv2.waitKey(5) & 0xFF == ord('r'):
            landmarks, landmarks_mir = record_landmarks(record_time, target)
            cv2.destroyAllWindows()  # Close the current window before opening the next one
            return landmarks, landmarks_mir

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# define empty array to save the data
all_samples = []
all_samples_mir = []

# Set iterations - less if we're just testing
iterations = None
if TESTING:
    iterations = range(5)
else:
    iterations = range(len(target_values))

# RECORD SAMPLES
for idx in iterations:
    current_samples, current_samples_mir = calibrate_gesture(target_values[idx], target_names[idx], RECORD_TIME)
    all_samples = all_samples + current_samples
    all_samples_mir = all_samples_mir + current_samples_mir


# Get timestamp for save file name
date_time = time.asctime().replace(' ', '_').replace(':', '')[4:] # return month_day_hh:mm:ss_year
date_time = date_time[-4:] + '_' + date_time[:-5] # turn into YYYY_MM_DD_hh:mm:ss

# Save gesture data to CSV
with open(path + f"dir_data_{ARM}_{date_time}.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['az'] + ['el'] + [i for i in range(num_landmarks)])
    for landmarks, gesture_label in all_samples:
        row = [gesture_label[0]] + [gesture_label[1]] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
        writer.writerow(row)

if AUG:
    # Save mirrored gesture data to CSV
    with open(path_mir + f"dir_data_{SWAP[ARM]}_{date_time}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['az'] + ['el'] + [i for i in range(num_landmarks)])
        for landmarks, gesture_label in all_samples_mir:
            row = [-gesture_label[0]] + [gesture_label[1]] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
            writer.writerow(row)

print("Calibration data saved successfully!")

cap.release()
cv2.destroyAllWindows()