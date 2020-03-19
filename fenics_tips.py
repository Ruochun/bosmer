from dolfin import *
from petsc4py import PETSc
from mshr import *
import scipy.interpolate as SPI 
import scipy.spatial as SPS
import numpy as np
from ctypes import *

def point_in_hull(point, hull, tolerance=1e-12):
    return all( (np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def backtrack_search(alpha, c_1, t0, t1, d0, d1, pos, tangent, point, lib, n=30, beta=0.5):
    KK = np.zeros((6,6))
    for i in range(n):
        new_t0 = t0+alpha*d0
        new_t1 = t1+alpha*d1
        new_t2 = 1.0 - new_t1 -  new_t0
        if new_t0<0. or new_t1<0. or new_t2<0.:
            alpha *= beta
            continue
        lib.shapeFunc2(2, np.array([new_t0,new_t1,new_t2]), weights, KK)
        if np.linalg.norm(pts.transpose()@KK[0,:]-point)<=np.linalg.norm(pos+c_1*alpha*(tangent@np.array([d0,d1]))-point):
            return alpha
        else:
            alpha *= beta
    raise Exception("no enough line search steps")

class ExpressionFromScipyFunction(Expression):
    def __init__(self, f, *args, **kwargs):
        self._f = f
        print(f)
    def eval(self, values, x):
        print(x)
        values[:] = self._f(*x)

print(list_krylov_solver_preconditioners())
print(PETSc.Options().getAll())

mesh =  UnitSquareMesh(30,30)
#File("./mesh/mesh.pvd") << mesh
BezElem = [[1,2,3,5,6,9],[7,8,9,4,5,1]]
BezElem = np.array(BezElem)
BezElem = BezElem - 1
BezPnt = [[0.0,0.0],[.5,0.0],[1.0,0.],[0.,.5],[.5,.5],[1.,.5],[0.,1.],[.5,1.],[1.,1.]]
BezPnt = np.array(BezPnt)
hulls = []
for Elem in BezElem:
    #print(BezPnt[Elem,:])
    hulls.append(SPS.ConvexHull(BezPnt[Elem,:]))

hull_number = []
for point in mesh.coordinates():
    #print(point)
    for i in range(len(hulls)):
        if point_in_hull(point, hulls[i]):
            hull_number.append(i)
            break

weights = np.array([1.,1.,1.,1.,1.,1.])
#weights = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
#print(hull_number)
libTIGA = CDLL("./c_lib/lib_tIGA.so")
libTIGA.shapeFunc2.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64)]
#libTIGA.shapeFunc2.restype = POINTER(POINTER(c_double))

T = np.zeros((len(mesh.coordinates()),2))
NN = np.zeros((6,6))
i = 0
for point in mesh.coordinates():
    t0, t1 = 0.3, 0.3
    pts = BezPnt[BezElem[hull_number[i]],:]
    #print("point=",point)
    for j in range(30):
        #print("t=",np.array([t0,t1,1.0-t0-t1]))
        libTIGA.shapeFunc2(2, np.array([t0,t1,1.0-t0-t1]), weights, NN) # N0=N, N1=dNx, N2=dNy, N3=ddNxx, N4=ddNyy, N5=ddNxy
        N = NN[0,:]
        dN = NN[1:3,:].transpose() # remember python 1:3 means 1~2
        pos = pts.transpose() @ N
        J = pts.transpose() @ dN
        J0 = np.linalg.det(J)
        dNdx = dN @ np.linalg.inv(J)
        R = pos - point
        #print("pos=",pos)
        #print("residual=",np.linalg.norm(R))
        if np.linalg.norm(R)<1e-9:
            break
        tangent = pts.transpose()@dN
        delta = np.linalg.solve(tangent, -R)
        delta = delta/np.linalg.norm(delta)  # looks important
        #step_size = np.linalg.norm(R)
        step_size=backtrack_search(np.linalg.norm(R), 0.5, t0, t1, delta[0], delta[1], pos, tangent, point,libTIGA)
        #print("alpha=",step_size)
        t0 += step_size*delta[0]
        t1 += step_size*delta[1]
    #raise Exception("stop")
    #print("++++++++")
    T[i,:] = [t0,t1]
    i += 1

#print(T)

# now reproduce the mesh
new_coord = np.zeros((len(mesh.coordinates()),2))
i = 0
for t in T:
    pts = BezPnt[BezElem[hull_number[i]],:]
    libTIGA.shapeFunc2(2, np.array([t[0],t[1],1.0-t[0]-t[1]]), weights, NN)
    pos = pts.transpose() @ NN[0,:]
    new_coord[i,:]=pos
    i+=1
print(new_coord)

"""
mesh = Mesh("10cyl.xml")
#mesh =  BoxMesh(Point(0,0,0),Point(1,1,1),2,1,1)
V = FunctionSpace(mesh, FiniteElement("CG", mesh.ufl_cell(), 1))
#cell = Cell(mesh,1)
#info("{:g}".format(cell.inradius()))
centers = []
rad = []
for cell in cells(mesh):
    #print(cell.get_vertex_coordinates())
    #info("{:g}, {:g}, {:g}, {:g}".format(cell.get_vertex_coordinates()[0],cell.get_vertex_coordinates()[1],cell.get_vertex_coordinates()[2],cell.inradius()))
    coord = np.average(np.array(cell.get_vertex_coordinates()).reshape((-1,3)), axis=0)
    #cell_stiff.append(np.hstack( (coord, cell.inradius()) ))
    centers.append(coord)
    rad.append(cell.inradius())
coord = np.array(centers)
values = np.array(rad)
print(len(values))
#x, y, z, values = data[:,0], data[:,1], data[:,2], data[:,3]
#interpolant = SPI.interp2d(x, y, values, kind='linear', copy=False, bounds_error=True)
#expression = ExpressionFromScipyFunction(interpolant, degree=2, domain=mesh)
interpolant = SPI.LinearNDInterpolator(coord, values, fill_value=np.mean(values))
v = mesh.coordinates()
nodal = interpolant(v)

f = Function(V)
f.vector()[:] = np.divide(1.0,np.power(nodal[dof_to_vertex_map(V)],2))
File("./mesh/test_cell.pvd") << f
#expression = ExpressionFromScipyFunction(interpolant, degree=1, domain=mesh)
#print(interpolant([1,.5,.5]))
#f.interpolate(expression)
#f = interpolate(expression, V)
"""
