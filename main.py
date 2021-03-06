import cv2 as cv
import numpy as np
from time import time, sleep
from math import atan2, sqrt, sin, cos, atan, tan, acos, asin
from numba import njit, prange, uint8
import serial
from crc import crc8
from dataclasses import dataclass

CAM_INDEX = 0
RESIZE = 1.5
MIN_OBJ_AREA = [15, 100, 100]
IMG_SIZE = (640, 360)
CENTER = tuple(np.load('center.npy').tolist())

FIELD_SIZE = (220, 180)

PI = 3.141592
RAD2DEG = 180 / PI
DEG2RAD = PI / 180

INF = 1e42

@dataclass
class PointProj:
    proj: float
    dist: float

def vec2point(vec):
    return Point(cos(vec.dir), sin(vec.dir)) * vec.size

def point2vec(point):
    return Vec(point.size(), atan2(point.y, point.x))

def np2point(arr):
    return Point(arr[0], arr[1])

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
    	
    	msign = 1
    	if px < 0:
    	    msign = -1
    	
    	res = PointProj(proj=Point(px, py-intercept).size() * msign, dist=(Point(px, py) - self).size())
    	
    	return res

    def tuple(self):
        return (self.x, self.y)
    
    def __mul__(self, k):
        return Point(self.x * k, self.y * k)
    
    def __truediv__(self, k):
        return Point(self.x / k, self.y / k)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
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
    
    def copy(self):
        return Vec(self.size, self.dir)
    
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

class Line:
    def __init__(self, coef=1, intercept=1, vertical=None):
        self.coef = coef
        self.intercept = intercept
        self.vertical = vertical
    
    def atX(self, x):
        if self.vertical != None:
            return None
        return Point(x, x * self.coef + self.intercept)
    
    def atY(self, y):
        # y = x * coef + i
        # x = (y - i) / coef
        return Point((y - self.intercept) / self.coef, y)

def makeLine(p1, p2):
    if (p1.x == p2.x):
        return Line(vertical=p1.y)
    return Line((p1.y - p2.y) / (p1.x - p2.x), p1.y - (p1.y - p2.y) / (p1.x - p2.x) * p1.x)

class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        self.coef = (p1.y - p2.y) / (p1.x - p2.x)
        self.intercept = p1.y - coef * p1.x

    def intersection(self, other):
        if isinstance(other, Segment):
            x = (other.intercept - self.intercept) / (self.coef - other.coef)
            y = self.coef * x + self.intercept

            if  x >= min(self.p1.x, self.p2.x) and x <= max(self.p1.x, self.p2.x) and y >= min(self.p1.y, self.p2.y) and y <= max(self.p1.x, self.p2.y) and \
                    x >= min(other.p1.x, other.p2.x) and x <= max(other.p1.x, other.p2.x) and y >= min(other.p1.y, other.p2.y) and y <= max(other.p1.x, other.p2.y):
                return [Point(x, y)]
            else:
                return []

class Curve:
    def __init__(self, pos, r, fr=0, to=2*PI):
        self.pos = pos
        self.r = r
        self.fr = fr
        self.to = to


class PointSet:
    def __init__(self, points):
        self.points = np.array(points, dtype=np.float32)

        if self.points.shape[1] != 2:
            raise IndexError

    def distances(self, point):
        return np.sum(np.power(self.points - np.array([point.x, point.y]), 2), axis=1)

    def proj(self, point):
        return self.points[np.argmin(self.distances(point))]

    def intersection(self, other):
        if isinstance(other, Line):
            intercept = other.intercept
            coef = other.coef
            dsts = np.power(intercept + coef * self.points[:, 0], 2) + np.power(self.points[:, 1] - intercept / coef, 2)

            index = np.argmin(dsts)
            return self.points[index]
            
        elif isinstance(other, PointSet):
            minDist = INF
            minIndex = None

            for point in other.points:
                dsts = sef.distances(point)
                index = np.argmin(dsts)
                dist = dsts[index]

                if dist < minDist:
                    minDist = dist
                    minIndex = index

            return self.points[minIndex]
        
        raise TypeError

    def inArea(self, point):
        segment = Segment(point, Point(point.x + INF, point.y))
        intersectionCount = 0

        for i in range(len(self.points)):
            first = Point(*self.points[i - 1])
            second = Point(*self.points[i])

            intersection = segment.intersection(Segment(first, second))

            intersectionCount += len(intersection)

        if intersectionCount % 2 == 1:
            return True
        else:
            return False

FREQ = 1

def curveToPoints(pos, r, fr=0, to=2*PI):
    angle = np.linspace(fr, to, int(FREQ * (2 * PI * r) * ((to - fr) / (2 * PI))))

    res = np.array([np.cos(angle) * r + pos.x, np.sin(angle) * r + pos.y], dtype=np.float32)

    return np.transpose(res)

def lineToPoints(p1, p2):
    if p1.x != p2.x:
        #print(p1.x, p2.x, p1.y, p2.y)
        coef = (p1.y - p2.y) / (p1.x - p2.x)
        #print(coef)
        intercept = p1.y - coef * p1.x
        #print(intercept)
        x = np.linspace(p1.x, p2.x, int(FREQ * (p1 - p2).size()))
        res = np.array([x, x * coef + intercept], dtype=np.float32)
        #print(res)
        return np.transpose(res)
    else:
        res = np.array([[p1.x] * int(FREQ * abs(p1.y - p2.y)), np.linspace(p1.y, p2.y, int(FREQ * abs(p1.y - p2.y)))], dtype=np.float32)

        return np.transpose(res)

def makeOutArea():
    corner = Point(182 / 2, 121 / 2)

    area = lineToPoints(-corner, Point(corner.x, -corner.y))
    area = np.append(area, lineToPoints(Point( corner.x,      -corner.y), Point(corner.x, -corner.y + 26)), axis=0)
    area = np.append(area, lineToPoints(Point( corner.x,      -corner.y + 26), Point(corner.x - 15, -corner.y + 26)), axis=0)
    area = np.append(area, curveToPoints(Point(corner.x - 15, -corner.y + 26 + 15), 15, fr = PI, to = PI * 1.5), axis=0)
    area = np.append(area, lineToPoints(Point( corner.x - 30, -corner.y + 26 + 15), Point(corner.x - 30, corner.y - 26 - 15)), axis=0)
    area = np.append(area, curveToPoints(Point(corner.x - 15,  corner.y - 26 - 15), 15, fr = PI * 0.5, to = PI), axis=0)
    area = np.append(area, lineToPoints(Point( corner.x - 15,  corner.y - 26), Point(corner.x, corner.y - 26)), axis=0)
    area = np.append(area, lineToPoints(Point( corner.x,       corner.y - 26), corner), axis=0)

    area = np.append(area, lineToPoints(corner, Point(-corner.x, corner.y)), axis=0)
    area = np.append(area, lineToPoints(Point( -corner.x,       corner.y), Point(-corner.x, corner.y - 26)), axis=0)
    area = np.append(area, lineToPoints(Point( -corner.x,       corner.y - 26), Point(-corner.x + 15, corner.y - 26)), axis=0)
    area = np.append(area, curveToPoints(Point(-corner.x + 15,  corner.y - 26 - 15), 15, fr = 0, to = PI * 0.5), axis=0)
    area = np.append(area, lineToPoints(Point( -corner.x + 30,  corner.y - 26 - 15), Point(-corner.x + 30, -corner.y + 26 + 15)), axis=0)
    area = np.append(area, curveToPoints(Point(-corner.x + 15, -corner.y + 26 + 15), 15, fr = PI * 1.5, to = PI * 2), axis=0)
    area = np.append(area, lineToPoints(Point( -corner.x + 15, -corner.y + 26), Point(-corner.x, -corner.y + 26)), axis=0)
    area = np.append(area, lineToPoints(Point( -corner.x,      -corner.y + 26), -corner), axis=0)

    return area
    
def makeGoalArea(index=1):
    corner = Point(182 / 2, 121 / 2)
    
    if index == 0:
        area = curveToPoints(Point(corner.x - 17, -corner.y + 26 + 15), 15, fr = PI, to = PI * 1.5)
        area = np.append(area, lineToPoints(Point( corner.x - 34, -corner.y + 26 + 15), Point(corner.x - 34, corner.y - 26 - 15)), axis=0)
        area = np.append(area, curveToPoints(Point(corner.x - 17,  corner.y - 26 - 15), 15, fr = PI * 0.5, to = PI), axis=0)

    elif index == 1:
        area = curveToPoints(Point(-corner.x + 17,  corner.y - 26 - 15), 15, fr = 0, to = PI * 0.5)
        area = np.append(area, lineToPoints(Point( -corner.x + 34,  corner.y - 26 - 15), Point(-corner.x + 34, -corner.y + 26 + 15)), axis=0)
        area = np.append(area, curveToPoints(Point(-corner.x + 17, -corner.y + 26 + 15), 15, fr = PI * 1.5, to = PI * 2), axis=0)

    else:
        raise ValueError

    return area

class SimplifiedBall:
    def __init__(self):
         self.pos = Point(0, 0)
         self.vel = Point(0, 0)
    
    def updatePos(self, dt):
        self.pos += self.vel * dt

class Robot:
    def __init__(self):
        self.pos = Point(0, 0)
        self.vel = Point(0, 0)
        
        self.ball = SimplifiedBall()

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
    
    def updateModel(self):
        global world
        world.virtualRobot.update(Vec(self.vel, self.dir * DEG2RAD))
    
    def toBytes(self):  
        self.updateModel()
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

        d1 = int(((360 - self.dir)%360) * K)
        d2 = int((((360 - self.dir)%360) * K - d1) * 256)

        h1 = int(self.head * K)
        h2 = int((self.head * K - h1) * 256)

        msg = [0xBB, int(self.vel * 50), d1, d2, h1, h2, int(self.acc), self.flags]
        print(msg)
        msg.append(crc8(msg))

        return bytes(msg)
        
    def stopBytes(self):
        self.updateModel()
        
        msg = [0xBB, 0, 0, 0, 0, 0, 0, 0]
        msg.append(crc8(msg))
        
        return bytes(msg)

class FieldEngine:
    def __init__(self):
        self.outArea = PointSet(makeOutArea())
        self.enemyGoal = PointSet(makeGoalArea())
        self.ourGoal = PointSet(makeGoalArea(0))
        
        ### Change that if you need to choose goals
        self.centerPoint = Point(70, 0)
        self.goalieTarget = self.centerPoint.copy()
        
    def update(self, world):
        rawBall = world.ball.pos
        currentBall = world.ball.predict(0.35)
        previousTarget = self.goalieTarget.copy()
        distToBall = abs(world.ball.pos.y - world.robot.pos.y)
        distCoef = distToBall / 40 - 1
        
        if distCoef < 0: ### When ball is near
            distCoef = 0
        if distCoef > 1: ### When ball is far away
           distCoef = 1
        
        ### something wrong with direction calculation, try to move ball from right up to bottom down
        target = currentBall
        target = self.centerPoint * distCoef + target * (1 - distCoef)
        target = target * 0.8 + previousTarget * 0.2
        self.goalieTarget = np2point(self.ourGoal.proj(target))
        
        
class FieldPainter:
    def __init__(self):
        self.size = Point(FIELD_SIZE[0], FIELD_SIZE[1])
        self.center = self.size / 2   
        
        self.outArea = makeOutArea()[::5]
        self.ourGoal = makeGoalArea()[::5]
        self.enemyGoal = makeGoalArea(0)[::5]
        self.plotter = []
        #for p in self.outArea:
        #    pass#print(p)
        
        self.color = (50, 200, 40)
        self.ballColor = (20, 30, 180)
        self.ballTrajectoryColor = (70, 150, 140)
    
    def drawLine(self, p1, p2):
        self.plotter.append({"type":"line", "points":[p1,p2]})
    
    def drawCircle(self, p, r):
        self.plotter.append({"type":"circle", "center": p, "radius": r})
    
    def update(self, world, trajectories=True):
        self.image = np.zeros((int(self.size.y), int(self.size.x), 3), dtype=np.uint8)
        self.image[:] = self.color
        
        for p in self.outArea:
            self.image = cv.circle(self.image, (int(self.center.x + p[0]), int(self.center.y + p[1])), 2, (250, 250, 250), -1)
        
        for p in self.ourGoal:
            self.image = cv.circle(self.image, (int(self.center.x + p[0]), int(self.center.y + p[1])), 2, (100, 120, 255), -1)
        
        for p in self.enemyGoal:
            self.image = cv.circle(self.image, (int(self.center.x + p[0]), int(self.center.y + p[1])), 2, (120, 255, 100), -1)
        
        self.image = cv.circle(self.image, (world.virtualRobot.pos + self.center).int().tuple(), 10, (140, 140, 140), -1)
        
        self.image = cv.circle(self.image, (world.ball.pos + self.center).int().tuple(), 3, self.ballColor, -1)
        
        self.image = cv.circle(self.image, (self.center + world.fieldEngine.goalieTarget).int().tuple(), 3, (5, 5, 5), -1)
            
        for d in self.plotter:
            if d["type"] == "line":
                p1,p2 = d["points"]
                self.image = cv.line(self.image, (p1 + self.center).int().tuple(), (p2 + self.center).int().tuple(), (255, 0, 0), 2)
            elif d["type"] == "circle":
                self.image = cv.circle(self.image, (self.center + d["center"]).int().tuple(), d["radius"], (170, 61, 54), -1)
            else:
                raise TypeError
        
        self.plotter = []
        
        if trajectories:
            self.image = cv.line(self.image, (world.ball.pos + self.center).int().tuple(), (self.center + world.ball.predict(20)).int().tuple(), self.ballTrajectoryColor, 2)
            #print("From {}, Target: {}".format(str(world.ball.pos), str(world.ball.predict(20))))
        
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

mousePosition = Point(0, 0)

def mouseCallback(event, x, y, flags, param):    
    
    ### APPLY RESIZE
    x /= RESIZE
    y /= RESIZE
    
    x = int(x)
    y = int(y)
    
    global mousePosition
    mousePosition = Point(x, y)
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
    elif key == ord('c') or key == ord('C'):
        global CENTER
        CENTER = mousePosition.tuple()
        
        np.save('center.npy', CENTER)

    return False

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def comp(a):
    return -a[1]
#rr = 0

def nall(arr, value):
    for x in arr:
        if not (x == value):
            return False
            
    return True

def calculatePos(weights):
    global world, fieldData, STATE, GOALIER, STRIKER
    
    L = 90
    #global rr
    if True or weights[YELLOW_GOAL] == 0 or weights[BLUE_GOAL] == 0:
        ### CALC CURRENT POSITION OF ROBOT
        yellowDist = pix2cm(fieldData.v[YELLOW_GOAL].size, PIX2CM_GOAL)
        yellowAngle = ((fieldData.v[YELLOW_GOAL].dir + 360 - world.interface.angle) % 360) * DEG2RAD
        posOfYellow = np.array([-L, 0]) + np.array([cos(yellowAngle), sin(yellowAngle)]) * yellowDist
        yellow = Point(posOfYellow[0], posOfYellow[1])
    
        blueDist = pix2cm(fieldData.v[BLUE_GOAL].size - 7, PIX2CM_GOAL)
        blueAngle = ((fieldData.v[BLUE_GOAL].dir + 360 - world.interface.angle) % 360) * DEG2RAD
        posOfBlue = np.array([L, 0]) + np.array([cos(blueAngle), sin(blueAngle)]) * blueDist
        blue = Point(posOfBlue[0], posOfBlue[1])
     
        if weights[YELLOW_GOAL] + weights[BLUE_GOAL] != 0:
            world.robot.pos = ((yellow * weights[YELLOW_GOAL] + blue * weights[BLUE_GOAL]) / (weights[YELLOW_GOAL] + weights[BLUE_GOAL]))
        
        ballAngle = ((fieldData.v[BALL].dir + 360 - world.interface.angle) % 360) * DEG2RAD
        
        #print(world.robot.pos)
    else:
        d2 = pix2cm(fieldData.v[YELLOW_GOAL].size - 8, PIX2CM_GOAL)
        d1 = pix2cm(fieldData.v[BLUE_GOAL].size + 6, PIX2CM_GOAL)
        alpha = fieldData.v[BLUE_GOAL].dir - fieldData.v[YELLOW_GOAL].dir
       
        if alpha < 0:
            alpha += 360
        
        
        alpha *= DEG2RAD
        
        val = (d2 ** 2 + (2 * L) ** 2 - d1 ** 2) / (2 * d2 * 2 * L)
        if abs(val) > 1:
            val = sign(val)
        
        beta1 = acos(val)
        beta2 = asin(sin(alpha) * d1 / (2 * L))
        beta = (beta1 + beta2) / 2
        
        x1 = L - d2 * cos(beta)
        x2 = d1 * cos(PI - alpha - beta) - L
        x = (x1 + x2) / 2
        
        y1 = d1 * sin(PI - alpha - beta)
        y2 = d2 * sin(beta)
        y = (y1 + y2) / 2 
        
        #print(d1, d2)
        
        angle = -beta * RAD2DEG + fieldData.v[YELLOW_GOAL].dir
        #print(f"{angle} eq {world.interface.angle}")
        
        world.robot.pos = Point(x, y)
        
        ballAngle = ((-fieldData.v[BALL].dir + 360 - angle) % 360) * DEG2RAD
        
    ballDist = pix2cm(fieldData.v[BALL].size, PIX2CM_BALL)
    if weights[BALL] != 0:
        if nall(world.ball.oldPoints, world.ball.pos):
            world.ball.oldPoints = [world.ball.pos.copy()] * len(world.ball.oldPoints)
            
        world.ball.updatePosition(world.robot.pos - Point(cos(ballAngle), sin(ballAngle)) * ballDist)
        world.ball.updateVelocity()
        world.robot.ball.pos = world.ball.pos - world.robot.pos
        
    else:
        world.ball.velVec = Vec(0, 0)
        if world.robot.ball.pos.size() < 18:
            world.robot.ball.pos = Point(-5, 0)
        world.ball.pos = world.robot.pos + world.robot.ball.pos
        world.ball.oldPoints = [world.ball.pos.copy()] * len(world.ball.oldPoints)
        world.ball.current = world.ball.pos.copy()
     
    world.fieldEngine.update(world)
    

def detect(frame):
    global col, fieldData, world, CENTER
    
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
        angle = (atan2(relativeCentroid[0], relativeCentroid[1]) * RAD2DEG + 360) % 360
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
        self.oldPoints = [Point(0, 0)] * 7
        self.projSize = 0
        self.current = Point(0, 0)
        
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
        self.lastTime = time()
        
        lineReg = LinearRegression()
        lineReg.fit([[el.x] for el in self.oldPoints], [[el.y] for el in self.oldPoints])
        
        params = [p.proj(lineReg.coef_, lineReg.intercept_) for p in self.oldPoints]
        
        proj = np.array([x.proj for x in params])
        dist = np.array([x.dist for x in params])
        
        added = 0
        if proj[0] > proj[-1]:
            added = PI
        
        projSize = ((np.max(proj) - np.min(proj)) - np.mean(np.abs(dist)) * 2) / (len(self.oldPoints) - 2)
        projSize /= dt
        
        if projSize < 0:
            projSize = 0
        
        self.velVec = Vec(projSize, added + atan(lineReg.coef_[0]))
        
        self.current = self.predict(0.5) * 0.5 + self.current * 0.5
        #print("Current: {}".format(self.current))
    
    def predict(self, t):
        pos = self.pos.copy()
        vel = self.velVec.copy()
        
        if vel.size < 0:
            vel.size = -vel.size
            vel.dir = PI + vel.dir
        
        for i in range(int(t * 100)):
            pos = pos + vec2point(vel * 0.01)
            vel.size -= 0.35 # ~35 cm/s^2
            
            if vel.size < 0:
                vel.size = 0
                return pos
        return pos
        #return self.pos + vec2point(self.velVec * t)

class RobotModel():
    def __init__(self):
        self.vels = [Point(0,0)] * 5
        self.dt = 1/50 ### CHANGE THERE
        self.pos = Point(0, 0)
        self.hist = [Point(0,0)] * 5
    
    def update(self, vel):
        global world
        self.vels.pop(0)
        cvel = vec2point(vel) * self.dt * 100
        cvel.y = -cvel.y
        self.vels.append(cvel) ### cm/s
        self.pos = world.robot.pos.copy()
        for vel in self.vels:
           self.pos += vel
           
        self.hist.pop(0)
        self.hist.append(self.pos.copy())
        #print('------')
        #for v in self.hist:
        #    print(v)
        #print('------')
        
        #print(f'{self.hist[0]} eq {world.robot.pos} (current {self.pos})')

class World:
    def __init__(self, acc = 20):
        self.interface = RobotInterface()
        self.interface.acc = acc
        self.fieldEngine = FieldEngine()
        
        self.robot = Robot()
        self.ball = Ball()
        self.virtualRobot = RobotModel()
        
        
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
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
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
PIX2CM_BALL = [[0, 0], [15, 12.6], [20, 18], [25, 25], [30, 34.5], [35, 44.2], [40, 53.7], [45, 63], [50, 72.3], [55, 80.3], [60, 88.6], [65, 95], [70, 101], [75, 106], [80, 111.2], [85, 115], [90, 119], [95, 122], [100, 125], [105, 128], [110, 131], [115, 133], [120, 135], [125, 138], [225, 188]]

PIX2CM_GOAL = [[0, 0], [10, 30.3], [15, 42.2], [20, 52.4], [25, 61.7], [30, 72.8], [35, 78.6], [40, 87], [45, 92.4], [50, 96.6], [55, 101.9], [60, 108.6], [65, 112.8], [70, 116.2], [75, 120.5], [80, 123], [85, 125.2], [90, 127.4], [95, 130], [100, 131.5], [190, 160]]

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
GOALIER = 0
STRIKER = 1
STYLE = GOALIER
t=time()
connectStateTime = -1
blockedConnect = False
dd = 0
kickBlocked = False
kickState = -1
ballOuterRadius = 25
ballOuterArea = PointSet(curveToPoints(Point(0, 0), ballOuterRadius, fr=0, to=2*PI))
while 1:
    ### READ FRAME
    _, frame = cap.read()
    #frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    frame = frame[0:-10, 150:-130, :]
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    ### RECALCULATE FRAME SIZE
    H, W, _ = frame.shape
    
    ### DENOISE
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    
    ### READ SERIAL
    readSTM()
    
    #print(f'{world.robot.pos} eq {world.fieldEngine.goalieTarget})')
    ### ALGO LOGIC
    plotter = []
    if STYLE == GOALIER:
        delta = world.robot.pos - world.fieldEngine.goalieTarget
        delta.y = -delta.y
        delta = point2vec(delta)
        vel = delta.size / 30 - 0.1

        if vel > 0.8:
            vel = 0.8    
        if vel < 0:
            vel = 0
        else:
            vel += 0.2
    
        world.interface.vel = vel
        world.interface.dir = (RAD2DEG * delta.dir + 360) % 360
        
        delta2 = world.robot.pos - world.ball.pos
        toBall = (RAD2DEG * point2vec(delta2).dir + 360) % 360
        if toBall > 180:
             toBall -= 360
        
        if (world.ball.pos - world.robot.pos).size() < 30 and abs(toBall) < 45 and abs(world.ball.pos.x) < abs(world.robot.pos.x) and not kickBlocked:
            kickState = time()
            kickBlocked = True
        
        if time() - kickState < 0.3:
            d = world.robot.pos - world.ball.pos
            d.y = -d.y 
            world.interface.dir = (point2vec(d).dir * RAD2DEG + 360) % 360
            world.interface.vel = 1.2
        elif time() - kickState > 2:
            kickBlocked = False
            
    else:
        if world.ball.pos.x > world.robot.pos.x:
            toBall = point2vec(world.ball.pos - world.robot.pos)
            print("NOMRAL")
        else:
            print("HOH")
            toBall = point2vec(world.ball.pos - world.robot.pos + Point(15, 0))
            
            if abs(toBall.dir - PI) < 0.5:
                 toBall = point2vec(world.ball.pos - world.robot.pos)
        toBall.dir = -toBall.dir
        toBall.dir *= RAD2DEG
        toBall.dir += 180
        
        if toBall.dir > 180:
            toBall.dir -= 360
        
        
        moveDir = 50 * toBall.dir / toBall.size
        world.interface.dir = (moveDir + 360) % 360
        world.interface.vel = 0.3
        
        
        
        print(world.interface.dir)
        
        if onLine:
             pass
        #fieldPainter.drawCircle(target, 3)
        
        #_ = vec2point(Vec(1, -world.interface.dir * DEG2RAD))
        #world.interface.dir = (point2vec(_).dir * RAD2DEG + 360) % 360
        
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
    
    fps = 1 / (time() - t)
    t = time()
    output = cv.putText(output, str(int(fps * 10) / 10), (10, 330), cv.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 128), 2, cv.LINE_AA)
    
    output = cv.circle(output, CENTER, 4, (255, 0, 0), -1)
    
    #print(cm2pix(60, PIX2CM_BALL))#output = cv.line(output, ball.pos
    
    #print(world.ball.pos)
    
    
    ### SHOW FRAME
    imshow('frame', cv.resize(output, None, fx = RESIZE, fy = RESIZE))
    
    #print(world.robot.pos, world.ball.pos, (world.ball.velVec.dir * RAD2DEG, world.ball.velVec.size), world.ball.current)
    
    #### KEYS
    if detectKeys():
        break

cap.release()
cv.destroyAllWindows()
