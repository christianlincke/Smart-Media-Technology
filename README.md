# *turn your gestures into MIDI control messages*

## Quickstart
Run ***main.py*** to detect you gestures and send them via the MIDI port 
(you may need to adjust the port at the beginning of the script).
The current mapping is:

**MIDI CHANNEL**:   1

**MIDI CCs**: \
hand_left:      0, \
stretch_left    1, \
az_left         2, \
el_left         3, \
hand_right      4, \
stretch_right   5, \
az_right        6, \
el_right        7 

Then assign the MIDI control to the parameters you want to control in your DAW. Depending on your DAW/Synth,
***Helpers/midi_send_cc.py*** might be handy to generate dummy messages.

**For a more in depth guide, see the documentation and the headers/comments in the scripts.**