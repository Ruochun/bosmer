from dolfin import *

class yPeriodic(SubDomain):
	def __init__(self, tolerance=DOLFIN_EPS, mapFrom=0.0, mapTo=1.0):
		SubDomain.__init__(self)
		self.tol = abs(tolerance)
		self.lower = mapFrom
		self.shift_dist = (mapTo - mapFrom)
	def inside(self, x, on_boundary):
		return bool(near(x[1], self.lower) and on_boundary)
	def map(self, x, y):
		y[0] = x[0]
		y[1] = x[1] - self.shift_dist


