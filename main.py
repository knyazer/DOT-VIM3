import cv2 as cv
import numpy as np
from time import time, sleep
from math import atan2, sqrt, sin, cos, atan, tan
from numba import njit, prange, uint8
import serial
from crc import crc8
from dataclasses import dataclass

CAM_INDEX = 0
RESIZE = 1.5
MIN_OBJ_AREA = [15, 100, 100]
IMG_SIZE = (640, 360)

FIELD_SIZE = (220, 180)

PI = 3.141592
RAD2DEG = 180 / PI
DEG2RAD = PI / 180

@dataclass
class PointProj:
    proj: float
    dist: float

def vec2point(vec):
    return Point(cos(vec.dir), sin(vec.dir)) * vec.size

def point2vec(point):
    return Vec(point.size(), atan2(point.y, point.x))

class Point:
    def __init__(self, x = 0, y = 1):
        self.x = x
        self.y = y
        
    def int(self):
        return Point(int(round(self.x)), int(round(self.y)))
       
    def proj(self, coef, intercept):
    	normal = -1 / coef
    	
    	# y = coef * x + intercept
    	# intercept = y - coef * x
    	
    	normalIntercept = self.y - normal * self.x
    	
    	# y = coef1 * x + intercept1 = y = coef2 * x + intercept2
    	# coef1 * x + intercept1 = coef2 * x + intercept2
    	# (coef2 - coef1) * x = intercept1 - intercept2
    	# x = (intercept1 - intercept2) / (coef2 - coef1)
    	
    	px = (intercept - normalIntercept) / (normal - coef)
    	py = px * coef + intercept
    	
    	res = PointProj(proj=Point(px, py-intercept).size(), dist=(Point(px, py) - self).size())
    	
    	return res

    def tuple(self):
        return (self.x, self.y)
    
    def __mul__(self, k):
        return Point(self.x * k, self.y * k)
    
    def __truediv__(self, k):
        return Point(self.x / k, self.y / k)
    
    def __add__(self, a):
        try:
            return Point(self.x + a.x, self.y + a.y)
        except KeyError:
            return Point(self.x + a, self.y + a)
    
    def __neg__(self):
        return Point(-self.x, -self.y)
    
    def __sub__(self, a):
        return self + (-a)
    
    def size(self):
        return sqrt(self.x ** 2 + self.y ** 2)
    
    def toVec(self):
        return Vec(atan2(self.y, self.x) * RAD2DEG, self.size())
    
    def copy(self):
        return Point(self.x, self.y)
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
 
class Vec:
    def __init__(self, size = 1, angle = 0):
        self.size = size
        self.dir = angle
    
    def __add__(self, other):
        x1 = cos(self.dir * DEG2RAD) * self.size
        x2 = cos(other.dir * DEG2RAD) * other.size
        
        y1 = sin(self.dir * DEG2RAD) * self.size
        y2 = sin(other.dir * DEG2RAD) * other.size
        
        x = x1 + x2
        y = y1 + y2
        return Vec(sqrt(x * x + y * y), (RAD2DEG * atan2(y, x) + 360) % 360)
    
    def __mul__(self, other):
       return Vec(self.size * other, self.dir)
    
    def tuple(self):
        return (self.size, self.dir)

class FieldData:
    def __init__(self):
        self.p = [Point(), Point(), Point()]
        self.v = [Vec(1, 0), Vec(-1, 0), Vec(-1, 0)]

class Robot:
    def __init__(self):
        self.pos = Point(0, 0)
        self.vel = Point(0, 0)

class RobotInterface:
    def __init__(self):        
        self.vel = 0
        self.dir = 0
        self.head = 0
        self.targetHead = 0
        self.acc = 25
        self.flags = 0
        self.angle = 0
        self.qAngle = [0] * 5
        
    def angleUpdate(self, angle):
        self.qAngle.append(angle)
        self.angle = self.qAngle.pop(0)
    
    def toBytes(self):
        if abs(self.targetHead - self.head) > 180:
            signum = -1
        else:
            signum = 1
            
        delta = self.targetHead - self.head
        if abs(delta) > 5:
             delta = sign(delta) * 5
        
        self.head += delta * signum
        self.head %= 360
        
        
        K = 256 / 360

        d1 = int(self.dir * K)
        d2 = int((self.dir * K - d1) * 256)

        h1 = int(self.head * K)
        h2 = int((self.head * K - h1) * 256)

        msg = [0xBB, int(self.vel * 50), d1, d2, h1, h2, int(self.acc), self.flags]
        msg.append(crc8(msg))

        return bytes(msg)
        
    def stopBytes(self):
        msg = [0xBB, 0, 0, 0, 0, 0, 0, 0]
        msg.append(crc8(msg))
        
        return bytes(msg)
     
class FieldPainter():
    def __init__(self):
        self.size = Point(FIELD_SIZE[0], FIELD_SIZE[1])
        self.center = self.size / 2   
        
        self.color = (50, 200, 40)
        self.ballColor = (20, 30, 180)
        self.ballTrajectoryColor = (70, 150, 140)
        
    def update(self, world, trajectories=True):
        self.image = np.zeros((int(self.size.y), int(self.size.x), 3), dtype=np.uint8)
        self.image[:] = self.color
        
        self.image = cv.circle(self.image, (world.ball.pos + self.center).int().tuple(), 4, self.ballColor, -1)
        
        if trajectories:
            self.image = cv.line(self.image, (world.ball.pos + self.center).int().tuple(), (self.center + world.ball.predict(1)).int().tuple(), self.ballTrajectoryColor, 2)
        
    def show(self, state=None):
        if state != None:
           self.update(state)
        
        imshow("Field", self.image)

@njit(parallel=True, cache=False, fastmath=True)
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

@njit(uint8[:,:](uint8[:,:,:], uint8[:,:,:], uint8), parallel=True, cache=False, fastmath=True)
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

    global ipos, frame, STATE, CALIBRATION, STOP

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        print('Calibration in progress...')
        STATE = CALIBRATION
        ipos[0].x, ipos[0].y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        ### UPDATE MOUSE POSITION
        ipos[1].x, ipos[1].y = x, y

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        print('Calibration end...')
        STATE = STOP
        
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
    global col, calibrationState, STATE, PLAY, world

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
        if STATE != STOP:
            STATE = STOP
        else:
            print("PLAY")
            STATE = PLAY

    return False

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def comp(a):
    return -a[1]

def calculatePos(weights):
    global world, fieldData
    
    ### CALC CURRENT POSITION OF ROBOT
    yellowDist = pix2cm(fieldData.v[YELLOW_GOAL].size - 5, PIX2CM_GOAL)
    yellowAngle = fieldData.v[YELLOW_GOAL].dir * DEG2RAD
    posOfYellow = np.array([90, 0]) + np.array([cos(yellowAngle), sin(yellowAngle)]) * yellowDist
    yellow = Point(posOfYellow[0], posOfYellow[1])
    
    blueDist = pix2cm(fieldData.v[BLUE_GOAL].size - 12, PIX2CM_GOAL)
    blueAngle = fieldData.v[BLUE_GOAL].dir * DEG2RAD
    posOfBlue = np.array([-90, 0]) + np.array([cos(blueAngle), sin(blueAngle)]) * blueDist
    blue = Point(posOfBlue[0], posOfBlue[1])
    
    weights[BLUE_GOAL] /= 3
    
    if (weights[YELLOW_GOAL] + weights[BLUE_GOAL]) != 0:
        if (blue - world.robot.pos).size() > 40:
            weights[BLUE_GOAL] /= 5
        
        if (yellow - world.robot.pos).size() > 40:
            weights[YELLOW_GOAL] /= 5
            
        world.robot.pos = ((yellow * weights[YELLOW_GOAL] + blue * weights[BLUE_GOAL]) / (weights[YELLOW_GOAL] + weights[BLUE_GOAL]))
        
        
    ballDist = pix2cm(fieldData.v[BALL].size, PIX2CM_BALL)
    ballAngle = fieldData.v[BALL].dir * DEG2RAD
    
    if weights[BALL] != 0:
        world.ball.updatePosition(world.robot.pos + -Point(cos(ballAngle), sin(ballAngle)) * ballDist)
        world.ball.updateVelocity()
    

def detect(frame):
    global col, fieldData, world
    
    thresh = np.zeros(frame.shape[:2], dtype=np.uint8)
    output = np.copy(frame)
    weights = [0,0,0]
    
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
        i = 1
        maxCompIndex = 0
        wholeWeight = 0
        relativeCentroid = np.array([0,0], dtype=np.float64)
        label = np.zeros(thresh.shape[:2], dtype=np.uint8)
        
        stats = np.delete(stats, 0, 0)
        centroids = np.delete(centroids, 0, 0)
        
        data = [[centroids[i], stats[i, cv.CC_STAT_AREA], i + 1] for i in range(centroids.shape[0])]
         
        data.sort(key = comp)
        
        for i in range(len(data)):
            
            ### IGNORE IF AREA OF COMPONENT IS VERY SMALL
            if (data[i][1] < MIN_OBJ_AREA[state]):
                break
                
            label += np.array(np.uint8(np.equal(labels, data[i][2])) * 255, dtype=np.uint8)
            
            ### CALCULATE POSITION OF COMPONENT CENTROID
            relativeCentroid += (data[i][0] - CENTER) * data[i][1]
            wholeWeight += data[i][1]
        
        if wholeWeight == 0:
            continue
            
        weights[state] = wholeWeight
        
        relativeCentroid /= wholeWeight
        
        ### CALCULATE SIZE AND DIRECTION OF OBJECT VECTOR
        angle = (atan2(relativeCentroid[0], relativeCentroid[1]) * RAD2DEG + 360 - world.interface.angle) % 360
        dist = np.sqrt(np.sum(relativeCentroid ** 2))
        
        ### SAVE DATA
        fieldData.v[state] = Vec(dist, angle)
        
        output[:, :, state] *= (label == 0)
        output[:, :, state - 1] *= (label == 0)
    
    calculatePos(weights)
    
    return output
    
inSeq = False
seqData = []
def readSTM():
    global ser, inSeq, seqData, world, STATE, STOP, PLAY, onLine
    
    LEN = 4
    while ser.in_waiting:
        x = ser.read()[0]
        if not inSeq and x == 0xBB:
            inSeq = True
            seqData = [x]
    
        elif inSeq:
            if len(seqData) == LEN:
                if x == crc8(seqData):
                    world.interface.angleUpdate((seqData[1] + seqData[2] / 256) * 360 / 256)
                    if seqData[3] % 2 == 1:
                        if STATE == STOP:
                            STATE = PLAY
                        else:
                            STATE = STOP
                            
                    if (seqData[3] / 2) % 2 == 1:
                        onLine = time()
                    
                else:
                    print("incorrect STM data")
                
                inSeq = False
            elif len(seqData) < LEN:
                seqData.append(x)
            else:
                inSeq = False
    
    world.interface.flags = int(world.interface.flags / 2) * 2 + int(STATE != STOP)

savedt = {}
def imshow(name, image):
    global savedt
    if time() - savedt.get(name, 0) > 0.12:
        cv.imshow(name, image)
        savedt[name] = time()
        
def adduction(val):
    while val > 180:
        val -= 360
    while val < -180:
        val += 360
        
    return val

@njit(parallel=True, cache=False, fastmath=True)
def makeMask(angles):
    W = IMG_SIZE[0]
    H = IMG_SIZE[1]
    
    arr = np.zeros((H, W), dtype=np.uint8)
    centroid = np.array([H / 2, W / 2])
    
    angles = angles * DEG2RAD
    eps = 0.02
    
    for i in prange(W):
        for j in prange(H):
            for k in prange(3):
                if abs(atan2(j - centroid[0], i - centroid[1]) - angles[k]) < eps:
                    arr[j][i] = 1
    
    return arr

from sklearn.linear_model import LinearRegression

class Ball:
    def __init__(self):
        self.pos = Point()
        self.vel = Point()
        self.velVec = Vec()
        self.lastTime = -1
        self.oldPoints = [Point(0, 0)] * 5
        self.projSize = 0
        
        self.r = 0.035
    
    def updatePosition(self, pos):
        self.pos = pos.copy()
        self.oldPoints.append(pos.copy())
        self.oldPoints.pop(0)
    
    def updateVelocity(self):
        if self.lastTime == -1:
            self.lastTime = time()
            return
        
        dt = time() - self.lastTime
        
        lineReg = LinearRegression()
        lineReg.fit([[el.x] for el in self.oldPoints], [[el.y] for el in self.oldPoints])
        
        params = [p.proj(lineReg.coef_, lineReg.intercept_) for p in self.oldPoints]
        
        proj = np.array([x.proj for x in params])
        dist = np.array([x.dist for x in params])
        
        if proj[0] > proj[-1]:
            lineReg.coef_[0] = -lineReg.coef_[0]
        
        projSize = ((np.max(proj) - np.min(proj)) / len(self.oldPoints)) - np.mean(np.abs(dist)) * 2
        projSize /= dt
        
        if projSize < 0:
            projSize = 0
        
        self.velVec = Vec(projSize, atan(lineReg.coef_[0]))
        
        self.lastTime = time()
    
    def predict(self, t):
        return self.pos + vec2point(self.velVec * t)

class World:
    def __init__(self, acc = 20):
        self.interface = RobotInterface()
        self.interface.acc = acc
        
        self.robot = Robot()
        self.ball = Ball()
        
        
world = World()

#mask = makeMask(np.array([116, -114, -1]))
#mask  =  mask[32:-32, 165:-165]
#cv.imshow("mask", mask)
### CONSTANTS AND VARIABLES
# OBJECTS INDEXING
BALL = 0
YELLOW_GOAL = 1
BLUE_GOAL = 2

# COLOR DATA
col = [ColorDetector("Ball"), ColorDetector("Yellow goal"), ColorDetector("Blue goal")]

for i in range(len(col)):
    col[i].load()

# CALIBRATION
calibrationState = 0
onLine = 0

### STATE PARAMS
STOP         = 0
PLAY         = 1
CALIBRATION  = 2
LINE_BACK    = 3
LINE_TO_BALL = 4
LINE_TO_LINE = 5
LINE_STOP    = 6
KICK         = 7
KICK_BACK    = 8
LINE_GETTER  = 9
TO_CENTER = 10

STATE = STOP

# MOUSE POSITION
ipos = [Point(), Point()]

# SERIAL CONFIGURATION
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0, bytesize=8, stopbits=1)

import os
os.system('v4l2-ctl -c white_balance_temperature_auto=1;')

### VISION CONFIGURATION
cap = cv.VideoCapture(CAM_INDEX)
cap.set(cv.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
cap.set(cv.CAP_PROP_CONTRAST, 25)
cap.set(cv.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv.CAP_PROP_SATURATION, 100)
cap.set(cv.CAP_PROP_GAMMA, 100)
cap.set(cv.CAP_PROP_GAIN, 40)

sleep(2)
os.system('v4l2-ctl -c white_balance_temperature_auto=0;')

cv.namedWindow('frame', cv.WINDOW_GUI_NORMAL + cv.WINDOW_AUTOSIZE)
cv.setMouseCallback('frame', mouseCallback)
cv.createTrackbar('Color range', 'frame', 1, 50, updateDelta)

### FIELD INFO
fieldData = FieldData()
fieldPainter = FieldPainter()

### ROBOT PARAMS
PIX2CM_BALL = [[0, 0], [13, 21.3], [18, 25.4], [23, 29.6], [28, 37.4], [33, 47.9], [38, 56.8], [43, 68.9], [48, 79.8], [53, 88.2], [58, 96.8], [63, 103.1], [68, 109], [73, 114.5], [78, 119.1], [83, 121.8], [88, 123.7], [93, 125.2], [98, 127.4], [103, 130.1], [203, 190.1]]

PIX2CM_GOAL = [[0, 0], [10, 24], [15, 34.3], [20, 49.9], [25, 60.9], [30, 69.5], [35, 75.5], [40, 82.2], [45, 86.6], [50, 91.4], [55, 95.5], [60, 97.7], [65, 101.7], [70, 104.8], [75, 107.2], [80, 109.5], [85, 111.2], [90, 113.4], [95, 114.2], [100, 115.7], [160, 134]]

def pix2cm(x, data):
    i = 0
    while data[i][1] < x and i + 1 < len(data):
        i += 1
    
    p1, p2 = data[i - 1], data[i]
    
    return (p1[0] * abs(p2[1] - x) + p2[0] * abs(p1[1] - x)) / (p2[1] - p1[1])

def cm2pix(x, data):

    i = 0
    while data[i][0] < x and i + 1 < len(data):
        i += 1
    
    p1, p2 = data[i - 1], data[i]
    
    return (p1[1] * abs(p2[0] - x) + p2[1] * abs(p1[0] - x)) / (p2[0] - p1[0])

### MAIN CYCLE
while 1:
    ### READ FRAME
    _, frame = cap.read()
    frame = frame[50:-50, 170:-170, :]
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    ### RECALCULATE FRAME SIZE
    H, W, _ = frame.shape
    CENTER = (130, 142)
    
    ### DENOISE
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    
    ### READ SERIAL
    readSTM()
        
    if STATE == CALIBRATION:
        output = inCalibration(frame)
    if STATE != CALIBRATION:
        output = detect(frame)
    
    if STATE == STOP or STATE == CALIBRATION:
        ser.write(world.interface.stopBytes())
    else:
        ser.write(world.interface.toBytes())
    
    fieldPainter.show(world)
    
    #print("BALL: " + str(world.ball.pos.tuple()) + "\t" + "VEL:" + str(world.ball.velVec.tuple()))
    
    ### SOME ADDITIONAL INFO
    output = cv.putText(output, col[calibrationState].name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 128), 2, cv.LINE_AA)
    
    #print(cm2pix(60, PIX2CM_BALL))#output = cv.line(output, ball.pos
    
    ### SHOW FRAME
    imshow('frame', cv.resize(output, None, fx = RESIZE, fy = RESIZE))
    
    #### KEYS
    if detectKeys():
        break

cap.release()
cv.destroyAllWindows()
