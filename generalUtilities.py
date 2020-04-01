from dolfin import *
from mshr import *
import numpy as np

def definePeriodic(meshData, args, sys_name, mapFrom, mapTo):
    if meshData[sys_name]['topoDim'] == 2:
        return globals()[args.periodic+'Periodic'](mapFrom=mapFrom, mapTo=mapTo)
    elif meshData[sys_name]['topoDim'] == 3:
        return globals()[args.periodic+'Periodic3D'](mapFrom=mapFrom, mapTo=mapTo)

class yPeriodic(SubDomain):
    def __init__(self, tolerance=1e-6, mapFrom=0.0, mapTo=1.0):
        SubDomain.__init__(self)
        self.tol = abs(tolerance)
        #self.map_tolerance = tolerance
        self.lower = mapFrom
        self.shift_dist = (mapTo - mapFrom)
    def inside(self, x, on_boundary):
        return bool(near(x[1], self.lower) and on_boundary)
        #return x[1]<(self.lower+self.tol) and x[1]>(self.lower-self.tol) and on_boundary
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - self.shift_dist

class yPeriodic3D(SubDomain):
    def __init__(self, tolerance=1e-6, mapFrom=0.0, mapTo=1.0):
        SubDomain.__init__(self)
        self.tol = abs(tolerance)
        #self.map_tolerance = tolerance
        self.lower = mapFrom
        self.shift_dist = (mapTo - mapFrom)
    def inside(self, x, on_boundary):
        return bool(near(x[1], self.lower) and on_boundary)
        #return x[1]<(self.lower+self.tol) and x[1]>(self.lower-self.tol) and on_boundary
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - self.shift_dist
        y[2] = x[2]

def explicitAppendSide(loop, start, dirVec, L, res):
    x = start[0]
    y = start[1]
    x_incre = dirVec[0]*L/res
    y_incre = dirVec[1]*L/res
    for i in range(0,res+1):
        loop.append(Point(x+i*x_incre, y+i*y_incre))
    return 0

def quadratureRulesTri(rule, nGQ=3):
    if rule == 'gauss4x4':
        x = np.array([  [0.0571041961,  0.06546699455602246],
                        [0.2768430136,  0.05021012321401679],
                        [0.5835904324,  0.02891208422223085],
                        [0.8602401357,  0.009703785123906346],
                        [0.0571041961,  0.3111645522491480],
                        [0.2768430136,  0.2386486597440242],
                        [0.5835904324,  0.1374191041243166],
                        [0.8602401357,  0.04612207989200404],
                        [0.0571041961,  0.6317312516508520],
                        [0.2768430136,  0.4845083266559759],
                        [0.5835904324,  0.2789904634756834],
                        [0.8602401357,  0.09363778440799593],
                        [0.0571041961,  0.8774288093439775],
                        [0.2768430136,  0.6729468631859832],
                        [0.5835904324,  0.3874974833777692],
                        [0.8602401357,  0.1300560791760936]])
        w = np.array([  0.04713673637581137, 0.07077613579259895, 0.04516809856187617,
                        0.01084645180365496, 0.08837017702418863, 0.1326884322074010,    
                        0.08467944903812383, 0.02033451909634504, 0.08837017702418863,
                        0.1326884322074010, 0.08467944903812383, 0.02033451909634504,
                        0.04713673637581137, 0.07077613579259895, 0.04516809856187617,
                        0.01084645180365496])

        return x, w


