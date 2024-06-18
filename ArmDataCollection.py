import cv2
import mediapipe as mp
from Models import ArmModel
import numpy as np
import time
import csv

# Which arm should be trained?
ARM = 'left'

# assign mask to extract relevant landmarks
landmark_mask = ArmModel.landmarkMask(ARM)

# Directory to save the data to
path = f'arm_direction_data_{ARM}/'

# Global variable to set record time for each gesture in seconds
RECORD_TIME = 10

# Define how many landmarks we have
num_landmarks = len(landmark_mask)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

# Function to capture landmarks for a set duration
def record_landmarks(duration, pose_label):
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
            landmarks_list.append((landmarks[landmark_mask], pose_label))

        # flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display the image with instructions.
        elapsed_time = time.time() - start_time
        cv2.putText(frame, f"Recording {ARM} arm {pose_label}... {duration - int(elapsed_time)}s left. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Pose Gesture Calibration', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    return landmarks_list

# Data collection with automated recording
def calibrate_gesture(gesture_index, record_time):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # gestures corresponding to 0 (center) , 2 (right) -2 (left)
        gesture_name = ["center", "half right", "right", "left", "half left", ]

        # flip frame horizontally
        frame = cv2.flip(frame, 1)
        # Display instructions
        cv2.putText(frame, f"Show {ARM} arm {gesture_name[gesture_index]} and press 'r' to record. Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Pose Gesture recording', frame)

        if cv2.waitKey(5) & 0xFF == ord('r'):
            landmarks = record_landmarks(record_time, gesture_index / 4)
            cv2.destroyAllWindows()  # Close the current window before opening the next one
            return landmarks

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Recording samples
center_samples = calibrate_gesture(0, RECORD_TIME)
half_right_samples = calibrate_gesture(1, RECORD_TIME)
right_samples = calibrate_gesture(2, RECORD_TIME)
left_closed_samples = calibrate_gesture(-2, RECORD_TIME)
half_left_samples = calibrate_gesture(-1, RECORD_TIME)

# Combine all samples for calibration and training
all_samples = center_samples + half_right_samples + right_samples + left_closed_samples + half_left_samples

# Get timestamp for save file name
date_time = time.asctime().replace(' ', '_')[4:] # return month_day_hh:mm:ss_year
date_time = date_time[-4:] + '_' + date_time[:-5] # turn into YYYY_MM_DD_hh:mm:ss

# Save gesture data to CSV
with open(path + f"arm_direction_data_{ARM}_{date_time}.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gesture'] + [f'[{i}]' for i in landmark_mask])
    for landmarks, gesture_label in all_samples:
        row = [gesture_label] + [f'{x},{y},{z}' for (x, y, z) in landmarks]
        writer.writerow(row)

print("Calibration data saved successfully!")

# Keep showing the screen
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the hands.
    results = pose.process(image)

    # Draw hand landmarks.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Data saved succesfully!. Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Pose Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()