from dolfin import *
import scipy.interpolate as SPI
import scipy.spatial as SPS
import scipy.io as IO
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import numpy.matlib 
from ctypes import *
import warnings

import generalUtilities as gU

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
    mesh['elemNode'] = mesh['elemNode'].astype(np.uint32) - 1 # because matlab
    mesh['bndNode'] =mesh['bndNode'].astype(np.uint32) - 1
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

def solveTIGALE2D(meshData, sys_name, problem, Var):
    E = 1.
    nu = .3
    mu = E/(2.*(1.+nu))
    lam = E*nu/((1.+nu)*(1.-2.*nu))
    D = problem['D']

    topoDim = meshData[sys_name]['topoDim']
    d = meshData[sys_name]['bzMesh']['degree']
    nen = int((d+1)*(d+2)/2)
    bzElem = meshData[sys_name]['bzMesh']['elemNode']
    bzCP = meshData[sys_name]['bzMesh']['cp']
    bnd = meshData[sys_name]['bzMesh']['bndNode']
    x, w = gU.quadratureRulesTri('gauss4x4')
    x = np.hstack((x, 1.0 - np.sum(x, axis=1)[:,None]))
    NN = np.empty((6, len(bzElem[0,:])))
    eldof = topoDim*len(bzElem[0,:])
    nTotTri = len(bzElem)*eldof**2
    row = np.empty(nTotTri).astype(np.uint32)
    col = np.empty(nTotTri).astype(np.uint32)
    val = np.zeros(nTotTri) 

    B = np.empty((eldof, topoDim))
    BT = np.empty((eldof, topoDim))
    trB = np.zeros((eldof, topoDim))
    ntriplets = 0
    for i in range(len(bzElem)):
        idx = bzElem[i,:]
        pts = bzCP[idx,:topoDim]
        wt = bzCP[idx,-1]
        dofidx = np.zeros(eldof)
        dofidx[0::topoDim] = topoDim*idx
        dofidx[1::topoDim] = topoDim*idx+1
        row[ntriplets:ntriplets+eldof**2] = np.matlib.repmat(dofidx, 1, eldof)
        col[ntriplets:ntriplets+eldof**2] = np.kron(dofidx, np.ones(eldof))
        for j in range(len(w)):
            libTIGA.shapeFunc2(d, x[j,:], wt, NN)
            N = NN[0,:]
            dN = NN[1:1+topoDim,:].transpose()
            J = pts.transpose() @ dN
            J0 = np.linalg.det(J)
            dNdx = dN @ np.linalg.inv(J)
            """
            B[0::topoDim,:] = dNdx
            B[1::topoDim,:] = dNdx
            np.copyto(BT,B)
            for k in range(nen):
                BT[2*k,1], BT[2*k+1,0] = B[2*k+1,0], B[2*k,1]
                trB[2*k,0], trB[2*k+1,1] = B[2*k,0] + B[2*k+1,1], B[2*k,0] + B[2*k+1,1]
            eleK = (B @ (mu*(B+BT) + lam*trB).transpose())*J0*w[j] 
            """
            B = np.zeros((3, eldof))
            B[0,0::2] = dNdx[:,0].transpose()
            B[1,1::2] = dNdx[:,1].transpose() 
            B[2,0::2] = dNdx[:,1].transpose()
            B[2,1::2] = dNdx[:,0].transpose()
            eleK = B.transpose() @ D @ B *J0*w[j]
            val[ntriplets:ntriplets+eldof**2] = val[ntriplets:ntriplets+eldof**2] + eleK.flatten('F')
        ntriplets += eldof**2
    K = sparse.coo_matrix((val, (row, col)))
    K = sparse.csr_matrix(K)
    del val, row, col
    F = np.zeros(topoDim*len(bzCP))

    bndNodes = np.unique(2*bnd.flatten())
    bndNodes = np.concatenate((bndNodes, bndNodes+1))
    #bndNodes = np.array([0,1,2,3]).astype(int)
    uD = np.zeros(len(bndNodes))
    uD[np.array([0,1,2,3])] = 0.1
   
    F = F - K[:,bndNodes]@uD
    F[bndNodes] = uD
    K[:,bndNodes] = 0.0
    K[bndNodes,:] = 0.0
    K[bndNodes, bndNodes] = 1.
    disp = spsolve(K, F)
    print(disp)
    meshData[sys_name]['bzMesh']['cp'][:,0] += disp[0::2]
    meshData[sys_name]['bzMesh']['cp'][:,1] += disp[1::2]
    IO.savemat('out.mat',meshData[sys_name]['bzMesh'])
    raise Exception

