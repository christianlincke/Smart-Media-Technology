"""
Test script to check if midi works.
"""
import mido
import rtmidi
import time

#set midi out port
port = mido.open_output('IAC-Treiber Bus 1')
#notes = [60, 62, 64, 65, 67, 69, 71, 72] # C Major Scale

while True:
    for i in range (127):
        msg = mido.Message('control_change', channel=1, control=1, value=i)
        port.send(msg)
        time.sleep(0.01)
        print('-' * i)
"""    for n in notes:
        msg = mido.Message('note_on', note=n)
        port.send(msg)

        time.sleep(0.5) # wait 0.5 seconds

        msg = mido.Message('note_off', note=n)
        port.send(msg)
"""

