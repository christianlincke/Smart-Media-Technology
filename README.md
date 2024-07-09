# README
(Quasi) functioning Model to convert gestures to MIDI.


### The structure of the scripts has changed significantly!!

#### xxDataCollection.py
still performs data collection (timestamp updated to also work on windows)

#### train.py
trains the models: parameter and side can be selected at the beginning of the script (ARM & PARAM)

#### main.py
performs detection for both sides (hand, stretch, az, el).
We dont have train data for all params yet! Until we have collected more data, set ARM = 'right' at the beginning of the script.

## Beware, a few naming conventions and formats have changed!
run /Helpers/mod_hand_csv.py to make sure your locally saved files still work! Newly recorded datasets will already use the new format
