import cv2 as cv
import numpy as np
from math import atan2, sqrt
import serial
from crc import crc8

FILE = 1

PI = 3.141592
RAD2DEG = 180 / PI

class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

    def tuple(self):
        return (self.x, self.y)

H = 360
W = 640
CENTER = (W / 2, H / 2)

frame = np.zeros((512, 512, 3), np.uint8)
ipos = [Point(), Point()]
calibrationInProgress = False
delta = np.array([10, 10, 10])
stateText = ['Ball', 'Yellow goal', 'Blue goal']


calibrationState = 0
ballCalibration, yellowGoalCalibration, blueGoalCalibration = [], [], []

if FILE:
    ballCalibration = np.load("save.npy").tolist()

def process(vel, direction, heading, acc):
    K = 256 / 360

    d1 = int(direction * K)
    d2 = int((direction * K - d1) * 256)

    h1 = int(heading * K)
    h2 = int((heading * K - h1) * 256)

    msg = [0xBB, int(vel * 10), d1, d2, h1, h2, int(acc * 10)]
    msg.append(crc8(msg))

    return msg

def readSTM():
    global ser
    if (ser.in_waiting):
        byte = ser.read()[0]
        if (byte & 15 != byte >> 4):
            print(byte & 15, byte >> 4)
            return None
        else:
            return (byte & 15) >> 1, (byte & 15) & 1
    return None

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0, bytesize=8, stopbits=1)


def mouseCallback(event, x, y, flags, param):
    x /= 2
    y /= 2
    
    x = int(x)
    y = int(y)
    
    global ipos, calibrationInProgress, frame

    if event == cv.EVENT_LBUTTONDOWN:
        print('Calibration in progress...')
        calibrationInProgress = True
        ipos[0].x, ipos[0].y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        ipos[1].x, ipos[1].y = x, y
        print(x,  y)

    elif event == cv.EVENT_LBUTTONUP:
        print('Calibration end...')
        calibrationInProgress = False
        calibrate(frame, ipos)

def calibrate(image, corner):
    global calibrationState, ballCalibration, yellowGoalCalibration, blueGoalCalibration
    if (corner[0].x == corner[1].x or corner[0].y == corner[1].y):
        return

    if (corner[0].x > corner[1].x):
        corner[1].x, corner[0].x = corner[0].x, corner[1].x

    if (corner[0].y > corner[1].y):
        corner[1].y, corner[0].y = corner[0].y, corner[1].y

    diap = image[ipos[0].y : ipos[1].y, ipos[0].x : ipos[1].x]

    calibrated = np.average(diap, axis=(0, 1))

    if calibrationState == 0:
        ballCalibration.append(calibrated)
    elif calibrationState == 1:
        blueGoalCalibration.append(calibrated)
    else:
        yellowGoalCalibration.append(calibrated)

def inCalibration(frame, alpha = 0.6):
    output = np.copy(frame)
    output = cv.rectangle(output, ipos[0].tuple(), ipos[1].tuple(), (128, 128, 0), -1)
    alpha = 0.6
    output = cv.addWeighted(output, alpha, frame, 1 - alpha, 0)

    return output

def detect(frame, color):
    global iangle, idist

    thresh = np.zeros(frame.shape[:2], dtype=np.bool)
    color = np.array(color)
    for calibValue in color:
        thresh |= cv.inRange(frame, calibValue - delta, calibValue + delta) == 255

    thresh = np.array(thresh, dtype=np.uint8) * 255

    if len(color) == 0:
        #print("No calibration")
        return frame, -1, -1
    if ((color == ballCalibration).all()):
        cv.imshow("thresh", thresh)
        
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)

    if len(stats) <= 1:
        print("No object on frame")
        return frame, -1, -1

    stats = np.delete(stats, 0, 0)

    maxCompIndex = np.argmax(stats[:, cv.CC_STAT_AREA], axis=None)
    label = np.array(np.uint8(np.equal(labels, maxCompIndex + 1)) * 255, dtype=np.uint8)
    relativeCentroid = centroids[maxCompIndex + 1] - CENTER
    
    if (stats[maxCompIndex, cv.CC_STAT_AREA] < 10):
        return frame, -1, -1
    
    angle = (atan2(relativeCentroid[0], relativeCentroid[1]) * RAD2DEG + 540) % 360
    dist = np.sqrt(np.sum(relativeCentroid ** 2))

    output = np.copy(frame)
    output[:, :, 0] *= (label == 0)
    output[:, :, 1] *= (label == 0)
    output[:, :, 2] *= (label == 0)

    return output, angle, dist

def setRange(val):
    global delta
    delta = np.array([val, val, val])

def detectKeys():
    global calibrationState

    key = cv.waitKey(1) & 0xFF

    if key == 27:
        return True
    elif key == ord('1'):
        calibrationState = 0
    elif key == ord('2'):
        calibrationState = 1
    elif key == ord('3'):
        calibrationState = 2

    return False


cap = cv.VideoCapture(0)

cv.namedWindow('frame')
cv.setMouseCallback('frame', mouseCallback)
cv.createTrackbar('Color range', 'frame', 10, 150, setRange)

vel = 1.0
direction = 0
heading = 0
acc = 10

import time

t1 = time.time()
fps = 100

def sign(x):
    if x > 0:
        return 1
    else:
        return -1
        
ballDist = 1
ballAngle = 0
while 1:
    _, frame = cap.read()
    frame = frame[50:-50, 150:-150, :]
    
    H, W, _ = frame.shape
    CENTER = (W / 2, H / 2)
    
    
    dt = time.time() - t1
    fps = 0.8 * fps + 0.2 / (time.time() - t1)
    t1 = time.time()
    
    
    # Main cycle
    if calibrationInProgress:
        output = inCalibration(frame)

    # Calibration
    else:
        outputlst = [0, 0, 0]
        outputlst[0], cBallAngle, cBallDist = detect(frame, ballCalibration)
        outputlst[1], blueGoalAngle, blueGoalDist = detect(frame, blueGoalCalibration)
        outputlst[2], yellowGoalAngle, yellowGoalDist = detect(frame, yellowGoalCalibration)

        output = outputlst[calibrationState]
        
        if (cBallDist != -1):
            ballDist = cBallDist
            
        if cBallAngle != -1:
            ballAngle = cBallAngle
        
        direction = (ballAngle + 360) % 360
        #print("!" + str(ballDist))
        
        direction -= 180
        
        if (abs(direction) > 20):
            direction += sign(direction) * 2200 / ballDist
        
        direction = (direction + 720) % 360
        
        print(ballAngle, ballDist)
        print(direction)
           

    output = cv.putText(output, stateText[calibrationState], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 128), 2, cv.LINE_AA)
    
    
    output = cv.putText(output, 'FPS: ' + str(int(fps)), (W - 170, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2, cv.LINE_AA)
    cv.imshow('frame', cv.resize(output, None , fx=2, fy = 2))
    
    #### STM
    val = readSTM()
    if (val != None):
        pass#print("STM says: \t" + str(val))

    ser.write(bytes(process(vel, direction, heading, acc)))

    #### KEYS    
    if detectKeys():
        break

if not FILE:
    np.save("save.npy", np.array(ballCalibration))


ser.write(bytes(process(0, direction, heading, acc)))

cap.release()
cv.destroyAllWindows()
