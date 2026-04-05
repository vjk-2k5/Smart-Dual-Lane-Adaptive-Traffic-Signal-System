import serial
import time

ser = serial.Serial('COM5', 9600)
time.sleep(2)

while True:
    ser.write(b"1\n")
    print("L1 Green")
    time.sleep(5)

    ser.write(b"7\n")
    print("All red (direct from green)")
    time.sleep(1.5)

    ser.write(b"6\n")
    print("L2 amber only")
    time.sleep(2)

    ser.write(b"3\n")
    print("L2 Green")
    time.sleep(5)

    ser.write(b"7\n")
    print("All red (direct from green)")
    time.sleep(1.5)

    ser.write(b"5\n")
    print("L1 amber only")
    time.sleep(2)
