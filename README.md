# README

I added the functionality to get data for both hands from one Collection run in ***3DDataCollection.py***.

Running ***/Helpers/plotLandmarksPose.py*** shows the results - **I am not quite sure if its doing the right thing!**
Therefore, for not please keep the setting of **TESTING = True** at the beginning of ***3DDataCollection.py*** until we know that we're not messing things up.

#### xxDataCollection.py
still performs data collection (timestamp updated to also work on windows)

#### train.py
trains the models: parameter and side can be selected at the beginning of the script (ARM & PARAM)

#### main.py
performs detection for both sides (hand, stretch, az, el).
We dont have train data for all params yet! Until we have collected more data, set ARM = 'right' at the beginning of the script.

## Beware, a few naming conventions and formats have changed!
run ***/Helpers/mod_hand_csv.py*** to make sure your locally saved ***hand_data_x.csv*** files still work! Newly recorded datasets will already use the new format
