import cv2
import mediapipe as mp
import torch
from Models import ArmModel3D
import mido

# Output MIDI? 'ON' / 'OFF'
MIDI = 'ON' # Turn Midi Output 'ON' or 'OFF'
midi_channel = 1 # MIDI Output Channel
midi_control_az = 1 # MIDI CC
midi_control_el = 2 # MIDI CC

# Which arm should be detected? 'left' or 'right'
ARM = 'right'

# assign mask to extract relevant landmarks
landmark_mask = ArmModel3D.landmarkMask(ARM)
num_landmarks = len(landmark_mask)

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')

# Initialize the model
model_path = 'Models/'
model = ArmModel3D.PoseGestureModel(in_feat=num_landmarks)
model.load_state_dict(torch.load(model_path + f'3D_model_{ARM}.pth'))
model.eval()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

def get_landmarks_coordinates(pose_landmarks):
    coordinates = []

    for i in landmark_mask:
        landmark = pose_landmarks.landmark[i]
        coordinates.extend([landmark.x, landmark.y, landmark.z])

    return coordinates


while cap.isOpened():
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

        # Get the landmark coordinates.
        landmarks = get_landmarks_coordinates(results.pose_landmarks)

        # Convert the landmarks to a tensor.
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

        # Predict the gesture.
        with torch.no_grad():
            prediction = model(input_tensor)
            direction_value = prediction[0] #.item()

        # Display the predicted spread value
        #print(f"Predicted spread value: {spread_value}")
    else:
        direction_value = [0.0, 0.0]


    # send midi cc message
    if MIDI == 'ON':
        cc_az = min(127, max(0, int((direction_value[0] + 0.5) * 127)))
        cc_el = min(127, max(0, int((direction_value[1] + 0.5) * 127)))
        msg_az = mido.Message('control_change', channel=midi_channel, control=midi_control_az, value=cc_az)
        msg_el = mido.Message('control_change', channel=midi_channel, control=midi_control_el, value=cc_el)
        port.send(msg_az)
        port.send(msg_el)
    else:
        cc_az = 'off'
        cc_el = 'off'

    # Flip & Display the image.
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Azimuth: {direction_value[0]:.2f} CC: {cc_az} Elevation: {direction_value[1]:.2f} CC: {cc_el}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Direction Detection. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Pose Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

