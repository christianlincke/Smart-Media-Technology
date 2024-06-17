"""
Hand Spread Recognition Script
Reads the hand landmarks using Media Pipe and evaluates them with the trained Model.
Outputs MIDI CC message for this value.
"""
import cv2
import mediapipe as mp
import torch
from Models import HandModel
import mido

# Define Midi thing
MIDI = 'ON' # Turn Midi Output 'ON' or 'OFF'
midi_channel = 1 # MIDI Output Channel
midi_control = 1 # MIDI CC Message

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')

# Initialize the model
model = handModel.HandGestureModel()
model.load_state_dict(torch.load('Models/hand_gesture_model.pth'))
model.eval()

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam.
cap = cv2.VideoCapture(0)

def get_landmarks_coordinates(hand_landmarks):
    coordinates = []
    for landmark in hand_landmarks.landmark:
        coordinates.extend([landmark.x, landmark.y, landmark.z])
    return coordinates


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the hands.
    results = hands.process(image)

    # Draw hand landmarks.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates.
            landmarks = get_landmarks_coordinates(hand_landmarks)

            # Convert the landmarks to a tensor.
            input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

            # Predict the gesture.
            with torch.no_grad():
                prediction = model(input_tensor)
                gesture_value = prediction.item()
    else:
        gesture_value = 0
        # Create and send midi cc message
    cc = min(127, max(0, int(gesture_value * 127)))

    if MIDI == 'ON':
        msg = mido.Message('control_change', channel=midi_channel, control=midi_control, value=cc)
        port.send(msg)

    # Display the predicted spread value
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, f"Gesture Value: {gesture_value:.2f} CC : {cc}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # print(f"Predicted spread value: {spread_value}")

    # Display the image.
    cv2.putText(frame, f"Hand Detection. Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

