import cv2
import mediapipe as mp
from Models import ArmModel
import numpy as np
import time
import csv

# Which arm should be trained?
ARM = 'right'  # or 'right'

# Global variable to set record time for each gesture in seconds
RECORD_TIME = 5

# stretch data is stored between fully stretched and fully bent. Indices 0 thru 4
target_values = [0.0, 0.25, 0.5, 0.75, 1.0]
target_names = ["100% bent", "75% bent", "50% bent", "75% straight", "100% straight", ]

# Directory to save the data to
path = f'TrainData/stretch_data_{ARM}/'

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

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose.
        results = pose.process(image)

        # Draw pose landmarks.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark])
            landmarks_list.append((landmarks, target))

        # flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display the image with instructions.
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording {ARM} arm {target}... {duration - int(elapsed_time)}s left. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Pose Gesture Calibration', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    return landmarks_list

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
            landmarks = record_landmarks(record_time, target)
            cv2.destroyAllWindows()  # Close the current window before opening the next one
            return landmarks

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# define empty array to save the data
all_samples = []

# RECORD SAMPLES
for idx in range(len(target_values)):
    current_samples = calibrate_gesture(target_values[idx], target_names[idx], RECORD_TIME)
    all_samples = all_samples + current_samples

# Get timestamp for save file name
date_time = time.asctime().replace(' ', '_').replace(':', '')[4:] # return month_day_hh:mm:ss_year
date_time = date_time[-4:] + '_' + date_time[:-5] # turn into YYYY_MM_DD_hh:mm:ss

# Save gesture data to CSV
with open(path + f"stretch_data_{ARM}_{date_time}.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gesture'] + [i for i in range(num_landmarks)])
    for landmarks, gesture_label in all_samples:
        row = [gesture_label] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
        writer.writerow(row)

print("Calibration data saved successfully!")

cap.release()
cv2.destroyAllWindows()