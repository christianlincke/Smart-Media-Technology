"""
perform both hand spread and arm direction or stretch detection.
Selection of arm, parameter, and midi setup can be done at the beginning of the script
last change: 19.06.2024 by christian
"""
import cv2
import mediapipe as mp
import torch
from Models import ArmModel, HandModel
import mido

# Which arm should be detected? 'left' or 'right' #
ARM = 'right' # 'left' or 'right'
PARAM = 'direction' # 'stretch' or 'direction'

# Define Midi stuff
MIDI = 'ON' # Turn Midi Output 'ON' or 'OFF'
MIDI_MODE = 'NOTE' # 'CC' or 'NOTE'
midi_channel = 1 # MIDI Output Channel
midi_control_hand = 1 # MIDI CC Message, if MIDI_MODE 'CC' is configured
midi_control_dir = 2 # MIDI CC Message, if MIDI_MODE 'CC' is configured
midi_note = 60 # MIDI Note to be send
midi_vel = 100 # MIDI velocity
midi_thresh = 0.5 # threshold at which the note triggers
dir_scale_factor = 14 # Factor to scale left - right range to one octave, only if MIDI_MODE 'NOTE'

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')

# define tracking variable, only if MIDI_MODE is note
if MIDI_MODE == 'NOTE':
    last_hand_value = 0

# assign mask to extract relevant landmarks
landmark_mask = ArmModel.landmarkMask(ARM)
num_landmarks_arm = len(landmark_mask)

# Initialize the models
model_path = 'Models/'
# Arm direction model
arm_model = ArmModel.PoseGestureModel(in_feat=num_landmarks_arm)     # Arm Model
arm_model.load_state_dict(torch.load(model_path + f'arm_{PARAM}_model_{ARM}.pth'))
arm_model.eval()
# Hand spread model
hand_model = HandModel.HandGestureModel()   # Hand Model
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

def check_hand_trigger(last_value, this_value, note):
    """
    checks if hand spread exceeds threshold and triggers note on/off.

    :param last_value: float hand value of the last frame
    :param this_value: float hand value of the current frame
    :param note: MIDI note
    :return:
    """

    if last_value < midi_thresh and this_value >= midi_thresh:
        # send Note On
        msg = mido.Message('note_on', channel=midi_channel, note=note, velocity=midi_vel)
        port.send(msg)
        print('note on!')
    elif last_value > midi_thresh and this_value <= midi_thresh:
        # send note off
        # to avoid notes getting stuck, we send note for all 127 notes. not elegant, to be optimized
        for n in range(128):
            msg = mido.Message('note_off', channel=midi_channel, note=n, velocity=midi_vel)
            port.send(msg)
        print('note off!')


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
        # print(f"Predicted spread value: {spread_value}")
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


    if MIDI == 'ON':
        if MIDI_MODE == 'NOTE':
            # send midi note (direction) when hand triggers
            midi_val_hand = int(min(1.999, max(0, int(hand_value * 2)))) # convert hand spread to (0, 1)
            midi_val_arm = int(min(72, max(60, (direction_value + 0.5) * dir_scale_factor + 60)))

            # check if hand triggers. send if trigger is detected
            check_hand_trigger(last_hand_value, hand_value, note=midi_val_arm)

            # Update value tracker for midi trigger
            last_hand_value = hand_value

        else:
            # send cc messages for hand and arm
            midi_val_hand = min(127, max(0, int(hand_value * 127))) # conver hand spread to midi range
            midi_val_arm = min(127, max(0, int((direction_value + 0.5) * 127))) # convers direction to midi range
            msg_hand = mido.Message('control_change', channel=midi_channel, control=midi_control_hand, value=midi_val_hand)
            msg_arm = mido.Message('control_change', channel=midi_channel, control=midi_control_dir, value=midi_val_arm)
            port.send(msg_hand)
            port.send(msg_arm)

    # Flip & Display the image.
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"{ARM} Hand and Arm detection. Press 'q' to quit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Hand Value: {hand_value:.2f} CC : {midi_val_hand}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Dir Value: {direction_value:.2f} CC : {midi_val_arm}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pose Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
