"""
perform 3d-direction, spread and hand detection for one arm
arm selection midi setup can be done at the beginning of the script
last change: 24.06.2024 by christian
"""
import cv2
import mediapipe as mp
import torch
from Models import ArmModel, ArmModel3D, HandModel
import mido

# Which arm should be detected? 'left' or 'right' #
ARM = 'right' # 'left' or 'right'

# Define Midi stuff
MIDI = 'ON' # Turn Midi Output 'ON' or 'OFF'
MIDI_MODE = 'CC' # 'CC' or 'NOTE'
midi_channel = 1 # MIDI Output Channel

midi_control_hand = 1 # MIDI CC Message, if MIDI_MODE 'CC' is configured
midi_control_stretch = 2 # MIDI CC Message, if MIDI_MODE 'CC' is configured

# set midi controls for azimuth and elevation
midi_control_az = 3
midi_control_el = 4

midi_note = 60 # MIDI Note to be send, currently not used
midi_vel = 100 # MIDI velocity
midi_thresh = 0.5 # threshold at which the note triggers, only used if MIDI_MODE is 'NOTE'
dir_scale_factor = 14 # Factor to scale to one octave, only used if MIDI_MODE is 'NOTE'

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')

# define tracking variable, only if MIDI_MODE is note
if MIDI_MODE == 'NOTE':
    last_hand_value = 0

# assign mask to extract relevant landmarks
landmarkMaskStretch = ArmModel.landmarkMask(ARM)
numLandmarksStretch = len(landmarkMaskStretch)

landmarkMaskDir = ArmModel3D.landmarkMask(ARM)
numLandmarksDir = len(landmarkMaskDir)

# Initialize the models
model_path = '../Models/'

# Arm 3D direction model
model_direction = ArmModel3D.PoseGestureModel(in_feat=numLandmarksStretch)     # Arm Model
model_direction.load_state_dict(torch.load(model_path + f'3D_model_{ARM}.pth'))
model_direction.eval()
# Arm stretch model
model_stretch = ArmModel.PoseGestureModel(in_feat=numLandmarksDir)     # Arm Model
model_stretch.load_state_dict(torch.load(model_path + f'arm_stretch_model_{ARM}.pth'))
model_stretch.eval()
# Hand spread model
model_hand = HandModel.HandGestureModel()   # Hand Model
model_hand.load_state_dict(torch.load('../Models/hand_gesture_model.pth'))
model_hand.eval()

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

def landmarks_to_coordinates_stretch(input_landmarks):
    coordinates = []

    for i in landmarkMaskStretch:
        landmark = input_landmarks.landmark[i]
        coordinates.extend([landmark.x, landmark.y, landmark.z])

    return coordinates

def landmarks_to_coordinates_direction(input_landmarks):
    coordinates = []

    for i in landmarkMaskDir:
        landmark = input_landmarks.landmark[i]
        coordinates.extend([landmark.x, landmark.y, landmark.z])

    return coordinates

def landmarks_to_coordinates_hand(input_landmarks):
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

    # Get arm stretch prediction and draw landmarks.
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the landmark coordinates.
        landmarks_conv_stretch = landmarks_to_coordinates_stretch(pose_results.pose_landmarks)
        landmarks_conv_dir = landmarks_to_coordinates_direction(pose_results.pose_landmarks)

        # Convert the landmarks to a tensor.
        input_tensor_stretch = torch.tensor(landmarks_conv_stretch, dtype=torch.float32).unsqueeze(0)
        input_tensor_dir = torch.tensor(landmarks_conv_dir, dtype=torch.float32).unsqueeze(0)

        # Predict stretch
        with torch.no_grad():
            prediction_stretch = model_stretch(input_tensor_stretch)
            value_stretch = prediction_stretch.item()

            prediction_direction = model_direction(input_tensor_dir)
            value_direction = prediction_direction[0] # .item()

        # Display the predicted spread value
        # print(f"Predicted spread value: {spread_value}")
    else:
        value_stretch = 0
        value_direction = [0, 0]

    # Get hand spread prediction and draw landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates.
            hand_landmarks_conv = landmarks_to_coordinates_hand(hand_landmarks)

            # Convert the landmarks to a tensor.
            input_tensor_hand = torch.tensor(hand_landmarks_conv, dtype=torch.float32).unsqueeze(0)

            # Predict the gesture.
            with torch.no_grad():
                prediction_hand = model_hand(input_tensor_hand)
                value_hand = prediction_hand.item()
    else:
        value_hand = 0

    # process raw data and generate MIDI messages
    if MIDI == 'ON':

        if MIDI_MODE == 'NOTE':
            # send midi note (stretch) when hand triggers
            midi_val_hand = int(min(1.999, max(0, int(value_hand * 2)))) # convert hand spread to (0, 1)
            midi_val_stretch = int(min(72, max(60, value_stretch * dir_scale_factor + 60)))

            # check if hand triggers. send if trigger is detected
            check_hand_trigger(last_hand_value, value_hand, note=midi_val_stretch)

            # Update value tracker for midi trigger
            last_hand_value = value_hand

        else:
            # send cc messages for hand and arm
            midi_val_hand = min(127, max(0, int(value_hand * 127))) # conver hand spread to midi range
            midi_val_stretch = min(127, max(0, int(value_stretch * 127))) # convers direction to midi range
            msg_hand = mido.Message('control_change', channel=midi_channel, control=midi_control_hand, value=midi_val_hand)
            msg_stretch = mido.Message('control_change', channel=midi_channel, control=midi_control_stretch, value=midi_val_stretch)
            port.send(msg_hand)
            port.send(msg_stretch)

        # always send azimuth and elevation data as CCs
        midi_val_az = min(127, max(0, int((value_direction[0] + 0.5) * 127)))
        midi_val_el = min(127, max(0, int((value_direction[1] + 0.5) * 127)))
        msg_az = mido.Message('control_change', channel=midi_channel, control=midi_control_az, value=midi_val_az)
        msg_el = mido.Message('control_change', channel=midi_channel, control=midi_control_el, value=midi_val_el)
        port.send(msg_az)
        port.send(msg_el)

    # Flip & Display the image.
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"{ARM} Hand and Arm detection. Press 'q' to quit.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Hand: {value_hand:.2f} CC : {midi_val_hand}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Stretch: {value_stretch:.2f} CC : {midi_val_stretch}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"AZ: {value_direction[0]:.2f} CC : {midi_val_az} EL : {value_direction[1]:.2f} CC : {midi_val_el}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pose Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
