import cv2 as cv
import numpy as np
from time import time
from math import atan2, sqrt
from numba import njit, prange, uint8
import serial
from crc import crc8

RESIZE = 1.5
MIN_OBJ_AREA = [40, 400, 400]
IMAGE_SIZE = 300

PI = 3.141592
RAD2DEG = 180 / PI
HIMAGE_SIZE = IMAGE_SIZE / 2

class Point:
    def __init__(self, x = 0, y = 1):
        self.x = x
        self.y = y

    def tuple(self):
        return (self.x, self.y)
 
class Vec:
    def __init__(self, size = 1, angle = 0):
        self.size = size
        self.dir = angle

class FieldData:
    def __init__(self):
        self.p = [Point(), Point(), Point()]
        self.v = [Vec(), Vec(), Vec()]

class Robot:
    def __init__(self):
        self.vel = 0
        self.dir = 0
        self.head = 0
        self.acc = 20
        self.flags = 0
    
    def toBytes(self):
        K = 256 / 360

        d1 = int(self.dir * K)
        d2 = int((self.dir * K - d1) * 256)

        h1 = int(self.head * K)
        h2 = int((self.head * K - h1) * 256)

        msg = [0xBB, int(self.vel * 50), d1, d2, h1, h2, int(self.acc * 10), self.flags]
        msg.append(crc8(msg))

        return bytes(msg)
        
    def stopBytes(self):
        msg = [0xBB, 0, 0, 0, 0, 0, 0, 0]
        msg.append(crc8(msg))
        
        return bytes(msg)
        

@njit(parallel=True, cache=True, fastmath=True)
def _CD_change(values, kernel, sz, m, d, data):
    x, y, z = 0, 0, 0
    stp = d * 2 + 1
    res = np.zeros((stp + sz - 1, stp + sz - 1, stp + sz - 1), dtype=np.uint8)

    if d != 0:
        res[d:-d, d:-d, d:-d] = data
    else:
        res = data

    for i in prange(values.shape[0]):
        x, y, z = values[i] >> m
        res[x : x + stp, y : y + stp, z : z + stp] = kernel

    if d != 0:
        data = res[d:-d, d: -d, d: -d]
    else:
        data = res

    return data

@njit(uint8[:,:](uint8[:,:,:], uint8[:,:,:], uint8), parallel=True, cache=True, fastmath=True)
def _CD_inRange(img, data, m):
    res = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            res[i][j] = data[img[i][j][0] >> m, img[i][j][1] >> m, img[i][j][2] >> m]

    return res

class ColorDetector:
    def __init__(self, name = "color", accuracy = 1):
        self.name = name
        self.m = accuracy
        self.sz = 256 >> self.m

        self.data = np.zeros((self.sz, self.sz, self.sz), dtype=np.uint8)

        self.updateDelta(4)

    def save(self):
        print('saved')
        np.save(self.name + ".npy", self.data)

    def load(self):
        print('loaded')
        self.data = np.load(self.name + ".npy")

    def updateDelta(self, d):
        self.d = d
        stp = d * 2 + 1
        self.removeKernel = np.zeros((stp, stp, stp), dtype=np.uint8)
        self.addKernel = self.removeKernel + 255

    def change(self, values, kernel):
        self.data = _CD_change(values, kernel, self.sz, self.m, self.d, self.data)

    def add(self, values):
        self.change(values, self.addKernel)

    def remove(self, values):
        self.change(values, self.removeKernel)

    def inRange(self, img):
        return _CD_inRange(img, self.data, self.m)


def mouseCallback(event, x, y, flags, param):
    ### APPLY RESIZE
    x /= RESIZE
    y /= RESIZE

    x = int(x)
    y = int(y)

    global ipos, frame, calibrationInProgress

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        print('Calibration in progress...')
        calibrationInProgress = True
        ipos[0].x, ipos[0].y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        ### UPDATE MOUSE POSITION
        ipos[1].x, ipos[1].y = x, y

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        print('Calibration end...')
        calibrationInProgress = False
        
        ### CALIBRATE
        if event == cv.EVENT_LBUTTONUP:
            calibrate(frame, ipos, True)
        else:
            calibrate(frame, ipos, False)

### MAIN CALIBRATION FUNC
def calibrate(image, corner, toAdd):
    ### MANAGE CORNERS
    if (corner[0].x == corner[1].x or corner[0].y == corner[1].y):
        return

    if (corner[0].x > corner[1].x):
        corner[1].x, corner[0].x = corner[0].x, corner[1].x

    if (corner[0].y > corner[1].y):
        corner[1].y, corner[0].y = corner[0].y, corner[1].y
    
    ### FIND PIXELS FOR CALIBRATION
    diap = image[ipos[0].y : ipos[1].y, ipos[0].x : ipos[1].x]
    diap = diap.reshape(diap.shape[0] * diap.shape[1], 3)

    global col, calibrationState
    
    ### UPDATE CALIBRATION DATA
    if toAdd:
        col[calibrationState].add(diap)
    else:
        col[calibrationState].remove(diap)

### DRAW RECT ON CALIBRATION ZONE
def inCalibration(frame, alpha = 0.6):
    global ipos

    output = np.copy(frame)
    output = cv.rectangle(output, ipos[0].tuple(), ipos[1].tuple(), (128, 128, 0), -1)
    alpha = 0.6
    output = cv.addWeighted(output, alpha, frame, 1 - alpha, 0)

    return output

def updateDelta(val):
    global col, calibrationState
    col[calibrationState].updateDelta(val)

def detectKeys():
    global col, calibrationState, playState, robot

    key = cv.waitKey(1) & 0xFF
    
    ### EXIT
    if key == 27:
        return True
    ### LOAD CALIBRATION
    elif key == ord('l') or key == ord('L'):
        col[calibrationState].load()
    ### SAVE CALIBRATION
    elif key == ord('s') or key == ord('S'):
        col[calibrationState].save()
    ### UPDATE CALIBRATION COLOR STATE
    elif key == ord('1'):
        calibrationState = BALL
    elif key == ord('2'):
        calibrationState = YELLOW_GOAL
    elif key == ord('3'):
        calibrationState = BLUE_GOAL
    elif key == ord('p') or key == ord('P'):
        playState = not playState

    return False

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def detect(frame):
    global col, fieldData
    
    thresh = np.zeros(frame.shape[:2], dtype=np.uint8)
    output = np.copy(frame)
    
    for state in [BALL, YELLOW_GOAL, BLUE_GOAL]:
        thresh = col[state].inRange(frame)
        
        ### SHOW CURRENT THRESH
        if state == calibrationState:
            imshow("thresh", thresh)
            
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)
        
        ### IGNORE IF THERE ARE NO OBJECTS
        if len(stats) <= 1:
            continue
        
        ### CALCULATE MAX COMPONENT SIZE EXCLUDE BACKGROUND
        stats = np.delete(stats, 0, 0)
        maxCompIndex = np.argmax(stats[:, cv.CC_STAT_AREA], axis=None)
        label = np.array(np.uint8(np.equal(labels, maxCompIndex + 1)) * 255, dtype=np.uint8)
        
        ### CALCULATE POSITION OF COMPONENT CENTROID
        relativeCentroid = centroids[maxCompIndex + 1] - CENTER
        
        ### IGNORE IF AREA OF COMPONENT IS VERY SMALL
        if (stats[maxCompIndex, cv.CC_STAT_AREA] < MIN_OBJ_AREA[state]):
            continue
        
        ### CALCULATE SIZE AND DIRECTION OF OBJECT VECTOR
        angle = (atan2(relativeCentroid[0], relativeCentroid[1]) * RAD2DEG + 540) % 360
        dist = np.sqrt(np.sum(relativeCentroid ** 2))
        
        print(angle)
        
        ### SAVE DATA
        fieldData.v[state] = Vec(dist, angle)
        
        output[:, :, state] *= (label == 0)
        output[:, :, state - 1] *= (label == 0)
        
    return output
    
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

savedt = {}
def imshow(name, image):
    global savedt
    if time() - savedt.get(name, 0) > 0.12:
        cv.imshow(name, image)
        savedt[name] = time()
        
def readSeq(x):
    global seqData, inSeq
    
    if not inSeq and x == 0xBB:
        inSeq = True
        seqData = []
    
    if inSeq:
        if len(seqData) == LEN:
            if x == crc8(seqData):
                print("correct")
            else:
                print("incorrect")
                
            inSeq = False
        else:
            seqData.append(x)
    
### CONSTANTS AND VARIABLES
# OBJECTS INDEXING
BALL = 0
YELLOW_GOAL = 1
BLUE_GOAL = 2

# COLOR DATA
col = [ColorDetector("Ball"), ColorDetector("Yellow goal"), ColorDetector("Blue goal")]

# CALIBRATION
calibrationInProgress = False
calibrationState = 0
playState = False
onLine = False

# MOUSE POSITION
ipos = [Point(), Point()]

# SERIAL CONFIGURATION
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0, bytesize=8, stopbits=1)

### VISION CONFIGURATION
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480);

cv.namedWindow('frame', cv.WINDOW_GUI_NORMAL + cv.WINDOW_AUTOSIZE)
cv.setMouseCallback('frame', mouseCallback)
cv.createTrackbar('Color range', 'frame', 4, 50, updateDelta)

### FIELD INFO
fieldData = FieldData()

### ROBOT PARAMS
robot = Robot()
robot.vel = 0.7

### MAIN CYCLE
while 1:
    ### READ FRAME
    _, frame = cap.read()
    frame = frame[32:-32, 165:-165, :]
    
    ### RECALCULATE FRAME SIZE
    H, W, _ = frame.shape
    CENTER = (W / 2, H / 2)
    
    ### DENOISE
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    
    ### CALIBRATION
    if calibrationInProgress:
        output = inCalibration(frame)
    ### IN WORK
    else:
        output = detect(frame)
        
        ballDist  = fieldData.v[BALL].size
        ballAngle = fieldData.v[BALL].dir
        
        direction = (ballAngle + 360) % 360
        
        direction -= 180
        
        if (abs(direction) > 20):
            direction += sign(direction) * 2000 / ballDist
        
        direction = (direction + 720) % 360
        
        robot.dir = direction
    
    ### READ SERIAL
    dataSTM = readSTM()
    while dataSTM != None:
        if dataSTM[0] == 0:
            playState = not playState
        if dataSTM[0] == 1:
            if (dataSTM[1] == 1):
                onLine = time()
                
        dataSTM = readSTM()
        
    robot.flags = int(robot.flags / 2) * 2 + int(playState)
    
    print(playState)
    if onLine + 0.1 > time():
        print("On line!!")
        direction = ((fieldData.v[YELLOW_GOAL].dir + fieldData.v[BLUE_GOAL].dir) / 2 + 180) % 360
        if abs(direction - fieldData.v[BALL].dir) % 360 < 90:
            robot.dir = direction
            
    else:
       print("Not on line")
    
    ### WRITE SERIAL
    if playState:
        ser.write(robot.toBytes())
    else:
        ser.write(robot.stopBytes())
    
    ### SOME ADDITIONAL INFO
    output = cv.putText(output, col[calibrationState].name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 128), 2, cv.LINE_AA)
    
    ### SHOW FRAME
    imshow('frame', cv.resize(output, None, fx = RESIZE, fy = RESIZE))
    
    #### KEYS
    if detectKeys():
        break

cap.release()
cv.destroyAllWindows()
