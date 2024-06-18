# README
(Quasi) functioning Model to convert gestures to MIDI.

#### main.py runs the hand & arm detection and outputs MIDI.

Each detection is compromised of four scripts:

#### ___DataCollection.py 
Collects data by recording landmarks for several different target  gestures. Input device is webcam. Data is saved as csv file

#### ___ModelTraining.py
Trains the Model with all acquired data. After training, weights are stored in /Models/___model.pth

#### Models/___Model.py
The actual NN with definitions for the Layers etc.

#### ___Detection.py
Script that converts the gestures to midi

### NEW
#### main.py
Arm and Hand detection merged into a single script!