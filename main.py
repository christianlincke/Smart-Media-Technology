"""
perform 3d-direction, spread and hand detection for one arm.
arm selection midi setup can be done at the beginning of the script
last change: 08.07.2024 by christian
"""
import cv2
import mediapipe as mp
import torch
from Models import ArmModel, ArmModel3D, HandModel
import mido
import copy
from google.protobuf.json_format import MessageToDict

# Which arm should be detected? 'left' or 'right'
ARM = 'right'  # 'left' or 'right' or 'both'

# left and right hand are reversed for some reason.
# Until we figure out why, this will fix it.
FLIP_HANDS = True

# Define Midi stuff
MIDI = 'ON'  # Turn Midi Output 'ON' or 'OFF'
MIDI_MODE = 'CC'  # 'CC' or 'NOTE'
MIDI_CHANNEL = 1  # MIDI Output Channel
MIDI_CONTROLS = {"hand_left":       0,
                 "stretch_left":    1,
                 "az_left":         2,
                 "el_left":         3,
                 "hand_right":      4,
                 "stretch_right":   5,
                 "az_right":        6,
                 "el_right":        7
                 }

MIDI_MAPPINGS = {"hand_left":       ((0, 1), (1, 127)),
                 "stretch_left":    ((0, 1), (1, 127)),
                 "az_left":         ((-0.5, 0.5), (1, 127)),
                 "el_left":         ((-0.5, 0.5), (1, 127)),
                 "hand_right":      ((0, 1), (1, 127)),
                 "stretch_right":   ((0, 1), (1, 127)),
                 "az_right":        ((-0.5, 0.5), (1, 127)),
                 "el_right":        ((-0.5, 0.5), (1, 127))
                 }

midi_note = 60  # MIDI Note to be sent, currently not used
midi_vel = 100  # MIDI velocity
MIDI_THRESH = 0.5  # threshold at which the note triggers, only used if MIDI_MODE is 'NOTE'
dir_scale_factor = 14  # Factor to scale to one octave, only used if MIDI_MODE is 'NOTE'

# set color and font for on screen text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTCOLOR = (0, 255, 255)

# define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Treiber Bus 1')


class Detector:

    def __init__(self, sides):
        """
        initialize variables, Models, cv2 etc
        """
        # list for sides to detect
        if sides.lower() == "left":
            self.sides = ["left"]
            num_hands = 1
        elif sides.lower() == "right":
            self.sides = ["right"]
            num_hands = 1
        elif sides.lower() == "both":
            self.sides = ["left", "right"]
            num_hands = 2
        else:
            raise ValueError(f"sides must be 'left', 'right' or 'both', not {sides}")


        # init variables
        self.frame = None

        # init detection values
        # utility
        self.value_names = ["hand_right", "stretch_right", "az_right", "el_right",
                            "hand_left", "stretch_left", "az_left", "el_left"]

        self.values_raw = {"hand_left": 0, "stretch_left": 0, "az_left": 0, "el_left": 0,
                           "hand_right": 0, "stretch_right": 0, "az_right": 0, "el_right": 0}

        # midi values, duplicate values_raw
        self.values_midi = copy.deepcopy(self.values_raw)

        # trigger memory values, duplicate values_raw
        self.values_prev = copy.deepcopy(self.values_raw)

        # initialize landmark masks
        self.__init_landmarkmasks()

        # initialize models
        self.__init_models()

        # initialize mediapipe
        self.__init_mp(num_hands)

        # Capture video from webcam.
        self.cap = cv2.VideoCapture(0)

    def __init_mp(self, num_hands):
        """
        initialize MediaPipe objects
        :return:
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True,
                                      min_detection_confidence=0.5)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=num_hands, min_detection_confidence=0.5)

        # init drawings
        self.mp_drawing = mp.solutions.drawing_utils

    def __init_landmarkmasks(self):
        """
        read landmark masks from model scripts
        :return:
        """
        self.landmark_masks = {"hand_left": HandModel.landmarkMask("left"),
                               "hand_right": HandModel.landmarkMask("right"),
                               "stretch_left": ArmModel.landmarkMask("left"),
                               "stretch_right": ArmModel.landmarkMask("right"),
                               "dir_left": ArmModel3D.landmarkMask("left"),
                               "dir_right": ArmModel3D.landmarkMask("right")}

        for param in ["stretch", "dir"]:
            if len(self.landmark_masks[f"{param}_left"]) != len(self.landmark_masks[f"{param}_right"]):
                raise ValueError(f"Landmark masks for {param} are not the same length!")

    def __init_models(self):
        """
        initialize models and load trained parameters
        :return:
        """
        # path to saved models
        model_path = 'Models/'

        # get number of in features for arm models
        num_landmarks_hand = len(self.landmark_masks["hand_left"])
        num_landmarks_stretch = len(self.landmark_masks["stretch_left"])
        num_landmarks_dir = len(self.landmark_masks["dir_left"])

        # initiate all models in a dict (easier to access sides)
        self.models = {"hand": HandModel.GestureModel(in_feat=num_landmarks_hand),
                       "stretch_right": ArmModel.GestureModel(in_feat=num_landmarks_stretch),
                       "stretch_left": ArmModel.GestureModel(in_feat=num_landmarks_stretch),
                       "dir_right": ArmModel3D.GestureModel(in_feat=num_landmarks_dir),
                       "dir_left": ArmModel3D.GestureModel(in_feat=num_landmarks_dir)
                       }

        # load stored model data and set mode to 'eval'
        self.models["hand"].load_state_dict(torch.load(model_path + 'hand_model.pth'))
        self.models["hand"].eval()

        for side in self.sides:
            self.models[f"stretch_{side}"].load_state_dict(torch.load(model_path + f'stretch_model_{side}.pth'))
            self.models[f"dir_{side}"].load_state_dict(torch.load(model_path + f'dir_model_{side}.pth'))
            self.models[f"stretch_{side}"].eval()
            self.models[f"dir_{side}"].eval()

    def set_device(self):
        """
        select gpu (if available) and move models
        :return:
        """

    def landmarks_to_coordinates(self, input_landmarks, param):
        """
        convert land marks from *.x *.y *.z to list
        also, apply landmark masks
        :param input_landmarks: list landmarks
        :param param: string which parameter {"hand", "stretch_left", "stretch_right", "dir_left", "dir_left"}
        :return: list converted and extracted coordinates
        """
        coordinates = []

        for i in self.landmark_masks[param]:
            landmark = input_landmarks.landmark[i]
            coordinates.extend([landmark.x, landmark.y, landmark.z])

        return coordinates

    def check_trigger(self, param, note):
        """
        checks if a value exceeds the threshold and triggers a not on
        :param param:
        :param note:
        :return:
        """

        if self.values_prev[param] < MIDI_THRESH and self.values_raw[param] >= MIDI_THRESH:
            # send Note On
            msg = mido.Message('note_on', channel=MIDI_CHANNEL, note=note, velocity=midi_vel)
            port.send(msg)
            print('note on!')
        elif self.values_prev[param] > MIDI_THRESH and self.values_raw[param] <= MIDI_THRESH:
            # send note off
            # to avoid notes getting stuck, we send note for all 127 notes. not elegant, to be optimized
            for n in range(128):
                msg = mido.Message('note_off', channel=MIDI_CHANNEL, note=n, velocity=midi_vel)
                port.send(msg)
            print('note off!')

    def eval_pose(self, pose_results):
        """
        evaluates the pose data with the NN

        :param pose_results:
        :return:
        """
        landmarks_conv = {"hand_left": 0, "stretch_left": 0, "dir_left": 0,
                          "hand_right": 0, "stretch_right": 0, "dir_right": 0}

        self.mp_drawing.draw_landmarks(self.frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Get the landmark coordinates and convert to tensor
        for side in self.sides:
            for param in ["hand", "stretch", "dir"]:

                landmarks_conv[f'{param}_{side}'] = self.landmarks_to_coordinates(pose_results.pose_landmarks, f'{param}_{side}')

                # Convert the landmarks to a tensor.
                landmarks_conv[f"{param}_{side}"] = torch.tensor(landmarks_conv[f"{param}_{side}"],
                                                                 dtype=torch.float32).unsqueeze(0)

        # Predict stretch & direction
        with torch.no_grad():
            for side in self.sides:

                prediction_stretch = self.models[f"stretch_{side}"](landmarks_conv[f"stretch_{side}"])
                self.values_raw[f"stretch_{side}"] = prediction_stretch.item()

                prediction_direction = self.models[f"dir_{side}"](landmarks_conv[f"dir_{side}"])
                self.values_raw[f"az_{side}"] = prediction_direction[0][0]
                self.values_raw[f"el_{side}"] = prediction_direction[0][1]

    def eval_hands(self, hand_results):
        """
        evaluates Hand data with the NN
        :param hand_results: mp landmarks
        :return:
        """
        # dict to swap "left" and "right"
        swap = {"left": "right", "right": "left"}

        # dict to store, which side has which index
        side_indexes = {0: None, 1: None}

        # iterate detected hands
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):

            # flip left and right
            if FLIP_HANDS:
                side_indexes[idx] = swap[hand_results.multi_handedness[idx].classification[0].label.lower()]
            else:
                side_indexes[idx] = hand_results.multi_handedness[idx].classification[0].label.lower()

            # skip iteration if hand should not be detected
            if side_indexes[idx] not in self.sides:
                continue

            # draw landmarks
            self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates
            hand_landmarks_conv = self.landmarks_to_coordinates(hand_landmarks, f"hand_{side_indexes[idx]}")

            # Convert the landmarks to a tensor.
            input_tensor_hand = torch.tensor(hand_landmarks_conv, dtype=torch.float32).unsqueeze(0)

            # Predict the gesture.
            with torch.no_grad():
                prediction_hand = self.models["hand"](input_tensor_hand)
                self.values_raw[f"hand_{side_indexes[idx]}"] = prediction_hand.item()

        # set values for hands that aren't detetcted to 0
        for side in ["left", "right"]:
            if side_indexes[0] != side and side_indexes[1] != side:
                self.values_raw[f"hand_{side}"] = 0

    def run(self):
        """
        open video, call the eval_*, send_midi and show_image methods for each frame
        :return:
        """

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Process the image and detect the pose and hands
            pose_results = self.pose.process(image_rgb)
            hand_results = self.hands.process(image_rgb)

            # Get arm stretch prediction and draw landmarks.
            if pose_results.pose_landmarks:
                self.eval_pose(pose_results)
            else:
                for side in self.sides:
                    self.values_raw[f"stretch_{side}"] = 0
                    self.values_raw[f"az_{side}"] = 0
                    self.values_raw[f"el_{side}"] = 0

            # Get hand spread prediction and draw landmarks
            if hand_results.multi_hand_landmarks:
                self.eval_hands(hand_results)
            else:
                self.values_raw["hand_right"] = 0
                self.values_raw["hand_left"] = 0

            # process raw data and generate MIDI messages
            if MIDI == 'ON':
                self.send_midi()

            # quit if 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # show current frame and values
            self.show_image()

        self.cap.release()
        cv2.destroyAllWindows()

    def map_values(self):
        """
        convert values from raw range to midi range
        y = m * x + b

        :return:
        """
        for val in self.value_names:
            # get ranges
            from_min = MIDI_MAPPINGS[val][0][0]
            from_max = MIDI_MAPPINGS[val][0][1]
            to_min = MIDI_MAPPINGS[val][1][0]
            to_max = MIDI_MAPPINGS[val][1][1]

            # calculate factor
            m = (to_max - to_min) / (from_max - from_min)
            # calculate offset
            b = to_min - (from_min * m)
            result = (m * self.values_raw[val]) + b

            # restrict range and convert to int
            self.values_midi[val] = int(min(to_max, max(to_min, result)))

    def send_midi(self):
        """
        convert values to MIDI range
        send midi messages
        :return:
        """
        # TODO add second arm processing
        
        if MIDI_MODE == 'NOTE':
            # send midi note (stretch) when hand triggers
            self.values_midi["hand_right"] = int(min(1.999, max(0, int(self.values_raw["hand_right"] * 2))))
            self.values_midi["stretch_right"] = int(min(72, max(60, self.values_raw["stretch_right"] * dir_scale_factor + 60)))

            # check if hand triggers. send if trigger is detected
            self.check_trigger("hand_right",
                               note=self.values_midi["stretch_right"])

            # Update value tracker for midi trigger
            self.values_prev = self.values_raw

        # always send CCs
        # map raw values to 0..127
        self.map_values()

        # send messages
        for val in self.value_names:
            msg = mido.Message('control_change', channel=MIDI_CHANNEL, control=MIDI_CONTROLS[val],
                               value=self.values_midi[val])
            port.send(msg)

    def show_image(self):
        """
        display the image and print useful data
        :return:
        """
        # Flip & Display the image.
        frame_flip = cv2.flip(self.frame, 1)

        cv2.putText(frame_flip, f"Hand and Arm detection. Press 'q' to quit.", (10, 30), FONT, 1, FONTCOLOR, 2)

        # print all values
        for idx, val in enumerate(self.value_names):
            pos = 30 * (idx+1) + 30
            cv2.putText(frame_flip, f"{val}:  {self.values_raw[val]:.2f}  {self.values_midi[val]:03d}",
                        (10, pos), FONT, 1, FONTCOLOR, 2)

        cv2.imshow('Pose Gesture Recognition', frame_flip)


if __name__ == "__main__":
    Det = Detector(ARM)
    Det.run()
