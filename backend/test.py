import serial
import time

ser = serial.Serial('COM5', 9600)
time.sleep(2)

while True:
    ser.write(b'1\n')
    print("L1 Green")
    time.sleep(5)

    ser.write(b'2\n')
    print("L1 Yellow")
    time.sleep(2)

    ser.write(b'3\n')
    print("L2 Green")
    time.sleep(5)

    ser.write(b'4\n')
    print("L2 Yellow")
    time.sleep(2)