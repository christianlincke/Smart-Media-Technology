"""
perform both hand spread and arm direction detection.
Select which arm should be detected in line 13.
17.06.2024
"""
import cv2
import mediapipe as mp
import torch
from Models import ArmModel, HandModel
import mido

# Which arm should be detected? 'left' or 'right' #
ARM = 'right'

# Output MIDI? 'ON' / 'OFF'
MIDI = 'ON' # Turn Midi Output 'ON' or 'OFF'
midi_channel = 1 # MIDI Output Channel
midi_control_hand = 1 # MIDI CC Message for hand spread
midi_control_dir = 2 # MIDI CC Message for arm direction

# assign mask to extract relevant landmarks
landmark_mask = ArmModel.landmarkMask(ARM)

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')

# Initialize the models
model_path = 'Models/'
# Arm direction model
arm_model = ArmModel.PoseGestureModel() # Arm Model
arm_model.load_state_dict(torch.load(model_path + f'arm_direction_model_{ARM}.pth'))
arm_model.eval()
# Hand spread model
hand_model = HandModel.HandGestureModel()
hand_model.load_state_dict(torch.load('Models/hand_gesture_model.pth'))
hand_model.eval()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

def arm_landmarks_coordinates(input_landmarks):
    coordinates = []

    for i in landmark_mask:
        landmark = input_landmarks.landmark[i]
        coordinates.extend([landmark.x, landmark.y, landmark.z])

    return coordinates

def hand_landmarks_coordinates(input_landmarks):
    coordinates = []

    for landmark in input_landmarks.landmark:
        coordinates.extend([landmark.x, landmark.y, landmark.z])
    return coordinates


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose and hands
    pose_results = pose.process(image)
    hand_results = hands.process(image)

    # Get arm direction prediction and draw landmarks.
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the landmark coordinates.
        pose_landmarks_conv = arm_landmarks_coordinates(pose_results.pose_landmarks)

        # Convert the landmarks to a tensor.
        input_tensor = torch.tensor(pose_landmarks_conv, dtype=torch.float32).unsqueeze(0)

        # Predict the gesture.
        with torch.no_grad():
            direction_prediction = arm_model(input_tensor)
            direction_value = direction_prediction.item()

        # Display the predicted spread value
        #print(f"Predicted spread value: {spread_value}")
    else:
        direction_value = 0

    # Get hand spread prediction and draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates.
            hand_landmarks_conv = hand_landmarks_coordinates(hand_landmarks)

            # Convert the landmarks to a tensor.
            input_tensor = torch.tensor(hand_landmarks_conv, dtype=torch.float32).unsqueeze(0)

            # Predict the gesture.
            with torch.no_grad():
                hand_prediction = hand_model(input_tensor)
                hand_value = hand_prediction.item()
    else:
        hand_value = 0

    # send midi cc message
    cc_hand = min(127, max(0, int((hand_value) * 127)))
    cc_dir = min(127, max(0, int((direction_value + 0.5) * 127)))
    if MIDI == 'ON':
        msg_hand = mido.Message('control_change', channel=midi_channel, control=midi_control_hand, value=cc_hand)
        msg_arm = mido.Message('control_change', channel=midi_channel, control=midi_control_dir, value=cc_dir)
        port.send(msg_hand)
        port.send(msg_arm)

    # Flip & Display the image.
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"{ARM} Hand and Arm detection. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Hand Value: {hand_value:.2f} CC : {cc_hand}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Dir Value: {direction_value:.2f} CC : {cc_dir}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pose Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

