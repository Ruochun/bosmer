from dolfin import *
import scipy.interpolate as SPI
import scipy.spatial as SPS
import scipy.io as IO
import numpy as np
from ctypes import *
import warnings

libTIGA = CDLL("./c_lib/lib_tIGA.so")
libTIGA.shapeFunc2.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64)]

def point_in_hull(point, hull, tolerance=1e-12):
    return all( (np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

def findPointInHull(pnt, hulls): # now the problem is in hull does not guarantee in element
    inlist = []
    for i in range(len(hulls)):
        if point_in_hull(pnt, hulls[i]):
            inlist.append(i)
            break
    return inlist
    
def initBzOrdinate(topoDim):
    if topoDim == 2:
        lucky = 0.3
        return np.array([lucky, lucky, 1.0-2*lucky])
    elif topoDim == 3:
        lucky = 0.2
        return np.array([lucky,lucky,lucky,1.0-3*lucky])

def backtrack_search_size(alpha, delta, degree, t, CPs, weights, pnt, guess, tangent, NN, c_1=0.5, beta=0.5, maxIter=20):
    new_t = np.empty(t.size)
    for i in range(maxIter):
        new_t[:-1] = t[:-1] + alpha*delta
        new_t[-1] = 1.0 - np.sum(new_t[:-1])
        if np.any(new_t<0.0):
            alpha *= beta
            continue
        libTIGA.shapeFunc2(degree, new_t, weights, NN)
        if np.linalg.norm(CPs.transpose()@NN[0,:]-pnt)<=np.linalg.norm(guess+c_1*alpha*(tangent@delta)-pnt):
            return alpha
        else:
            alpha *= beta
    raise Exception("Within {:d} trial iterations no proper step size found.".format(maxIter))

def findBzOrdinateViaOpt2D(pnt, d, CPs, weights, NN, maxIter=50, goodCrit=1e-7):
    t = initBzOrdinate(len(pnt))
    foundFlag = False
    for j in range(maxIter):
        libTIGA.shapeFunc2(d, t, weights, NN)
        N = NN[0,:]
        dN = NN[1:3,:].transpose()
        guess = CPs.transpose() @ N
        #J = pts.transpose() @ dN
        #J0 = np.linalg.det(J)
        #dNdx = dN @ np.linalg.inv(J)
        R = guess - pnt
        normR = np.linalg.norm(R)
        if normR<goodCrit:
            foundFlag = True
            break
        tangent = CPs.transpose() @ dN
        delta = np.linalg.solve(tangent, -R)
        delta = delta/np.linalg.norm(delta)
        step_size = backtrack_search_size(10*normR, delta, d, t, CPs, weights, pnt, guess, tangent, NN)
        t[:-1] += step_size*delta
        t[-1] = 1.0 - np.sum(t[:-1])
    return foundFlag, t, normR


def loadMatlabBezierMesh(args):
    mesh = IO.loadmat(args.tiga_mesh_file)
    mesh['elemNode'] = mesh['elemNode'] - 1 # because matlab
    return mesh

def formMapBezier2Lagrangian(bzMesh, lagMesh, topoDim):
    bzElem = bzMesh['elemNode']
    bzCP = bzMesh['cp']
    bzDeg = bzMesh['degree']
    # build convex hull info
    hulls = []
    for elem in bzElem:
        hulls.append(SPS.ConvexHull(bzCP[elem,:topoDim]))
    pntHullRegionInfo = np.zeros((len(lagMesh.coordinates()),2)).astype(int)
    #hull_number.fill(-1)
    i = 0
    for pnt in lagMesh.coordinates():
        pntHullRegionInfo[i,:] = findPointInHull(pnt, hulls)
        i += 1
    # opt to get mapping
    T = np.empty((len(lagMesh.coordinates()),1+topoDim)) # for each row: bz ordinate
    I = np.empty((len(lagMesh.coordinates()),1)).astype(int) # for each row: no. of beElem the node is in
    N = np.empty((6, len(bzElem[0,:]))) # dummy shape function variable
    i = 0
    for pnt in lagMesh.coordinates():
        for possibleBzElem in np.unique(pntHullRegionInfo[i,:]):
            CPs = bzCP[bzElem[possibleBzElem],:topoDim]
            weights = bzCP[bzElem[possibleBzElem],-1]
            foundThisPnt, T[i,:], R = findBzOrdinateViaOpt2D(pnt, bzDeg, CPs, weights, N)
            if foundThisPnt:
                I[i] = possibleBzElem
                break
        if not(foundThisPnt):
            I[i] = possibleBzElem
            warnings.warn('Lagrangian node no.{:d} probably has no proper position in Bezier mesh, the last iter residual is {:g}. Now proceed as is.'.format(i,R))
        i += 1
        
    return I, T    

def solveTIGALE(meshData, sys_name, problem):
    bzElem = meshData[sys_name]['bzMesh']['elemNode']
    bzCP = meshData[sys_name]['bzMesh']['cp']

    #for 
    


