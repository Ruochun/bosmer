from dolfin import *
from mshr import *

class yPeriodic(SubDomain):
	def __init__(self, tolerance=1e-6, mapFrom=0.0, mapTo=1.0):
		SubDomain.__init__(self)
		self.tol = abs(tolerance)
		self.lower = mapFrom
		self.shift_dist = (mapTo - mapFrom)
	def inside(self, x, on_boundary):
		return bool(near(x[1], self.lower) and on_boundary)
	def map(self, x, y):
		y[0] = x[0]
		y[1] = x[1] - self.shift_dist

def explicitAppendSide(loop, start, dirVec, L, res):
    x = start[0]
    y = start[1]
    x_incre = dirVec[0]*L/res
    y_incre = dirVec[1]*L/res
    for i in range(1,res+1):
        loop.append(Point(x+i*x_incre, y+i*y_incre))
    return 0

