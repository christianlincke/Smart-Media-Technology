# README
(Quasi) functioning Model to convert gestures to MIDI.

#### main.py runs the hand & arm detection and outputs MIDI.
## to use the stretch detetction, you will need to collect data first and thrain the model!

Each detection is compromised of four scripts:

#### ___DataCollection.py 
Collects data by recording landmarks for several different target  gestures. Input device is webcam. Data is saved as csv file
Hand data collection is straight forward.
For arm data collection, you need to select the hand (left/right) and the parameter (direction/stretch) you want to record.

#### ___ModelTraining.py
Trains the Model with all acquired data. After training, weights are stored in /Models/___model.pth
Hand Model training is straight forward.
For arm Model Training, you need to select the hand (left/right) and the parameter (direction/stretch) you want to train.

#### Models/___Model.py
The actual NN with definitions for the Layers etc.

#### ___Detection.py
Script that converts the gestures to midi

### main.py
Arm and Hand detection merged into a single script! At the beginning of the Script, the MIDI_MODE can be selected to send MIDI-CC o MIDI note-on. In the current Version, the arm values control the midi note and opening the hand triggers a note.

## 3D
very basic but seems to work. data collection needs to be adjusted.
