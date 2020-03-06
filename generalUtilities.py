from dolfin import *
from mshr import *

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

def explicitAppendSide(loop, start, dirVec, L, res, incFirst=True):
    x = start[0]
    y = start[1]
    x_incre = dirVec[0]*L/res
    y_incre = dirVec[1]*L/res
    if incFirst:
        s = 0
    else:
        s = 1
    for i in range(s,res+1):
        loop.append(Point(x+i*x_incre, y+i*y_incre))
    return 0

