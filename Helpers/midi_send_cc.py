"""
script sends midi cc on given channel and control
"""
import time
import mido

port = mido.open_output('IAC-Treiber Bus 1')


if __name__ == "__main__":

    channel = int(input("what channel (0 ... 15)\n"))
    control = int(input("what control [0 ... 127]\n"))

    for val in range(128):
        msg = mido.Message('control_change', channel=channel, control=control, value=val)
        port.send(msg)
        time.sleep(0.01)

    # send 0
    msg = mido.Message('control_change', channel=channel, control=control, value=0)
    port.send(msg)




