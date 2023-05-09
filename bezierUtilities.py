from dolfin import *
import scipy.interpolate as SPI
import scipy.spatial as SPS
import scipy.io as IO
import scipy.sparse as sparse
from scipy.special import comb
from scipy.sparse.linalg import spsolve
from scipy.interpolate import BSpline
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

def pyBernstein(degree, t):
    B = np.empty(degree+1).astype(float)
    for i in range(len(B)):
        B[i] = comb(degree, i)*(t**i)*(1.0-t)**(degree-i)
    return B

def pyDerBernstein(degree, t):
    A = np.zeros(degree+1).astype(float)
    B = np.zeros(degree+1).astype(float)
    A[1:] = pyBernstein(degree-1, t)
    B[:-1] = pyBernstein(degree-1, t)
    return degree*(A - B)

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
    mesh['bndNode'] = mesh['bndNode'].astype(np.int32)
    mesh['bndNode'][:,:-1] -= 1
    mesh['uniqueBnd'] = np.unique(mesh['bndNode'][:,:-1])
    mesh['degree'] = np.asscalar(mesh['degree'])
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
    I = np.empty(len(lagMesh.coordinates())).astype(int) # for each row: no. of beElem the node is in
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
        
    return T, I 

#def applyBzDirichletBC(K, F, bnd, BCs, Var):
def queryBCs2DFlavorLeastSquare(bnd, bzCP, BCs, topoDim, degree, lagFunc):
    uD = np.zeros(topoDim*len(bzCP))
    numSamples = 20 # degree + 1
    sampleU = np.linspace(0.0,1.0,num=numSamples)
    bzu = np.concatenate((np.zeros(degree + 1), np.ones(degree + 1)))
    xCP = np.empty(numSamples).astype(float)
    yCP = np.empty(numSamples).astype(float)
    evalBernsteinBasis = np.empty((numSamples,degree+1)).astype(float)
    for segment in bnd:
        if not(isinstance(BCs[segment[-1]], str)): # not str means it's numeric BC assignment, just directly enforce it
            uD[topoDim*segment[:-1]] = BCs[segment[-1]][0]
            uD[topoDim*segment[:-1]+1] = BCs[segment[-1]][1]
            continue
        pts = bzCP[segment[:-1],:topoDim]
        weights = bzCP[segment[:-1],-1] # current implementation uses BSpline package, weights in fact play no roles
        spl = BSpline(bzu, pts, degree)
        i = 0
        for samplePoint in spl(sampleU):
            xCP[i] = lagFunc[BCs[segment[-1]]](samplePoint)[0] # the Lagrangian query
            yCP[i] = lagFunc[BCs[segment[-1]]](samplePoint)[1]
            evalBernsteinBasis[i,:] = pyBernstein(degree, sampleU[i])
            i += 1
        uD[topoDim*segment[:-1]] = np.linalg.lstsq(evalBernsteinBasis, xCP, rcond=None)[0] # use only the first return
        uD[topoDim*segment[:-1]+1] = np.linalg.lstsq(evalBernsteinBasis, yCP, rcond=None)[0]

    return uD

def queryBCs2DFlavorLinearSystem(bnd, bzCP, BCs, topoDim, degree, lagFunc):
    uD = np.zeros(topoDim*len(bzCP))
    bzu = np.concatenate((np.zeros(degree + 1), np.ones(degree + 1)))
    x, w = gU.quadratureRulesLine('gauss', nGQ=degree//2+1)
    for i in range(len(bnd)):
        if not(isinstance(BCs[bnd[i,-1]], str)): # not str means it's numeric BC assignment, just directly enforce it
            uD[topoDim*bnd[i,:-1]] = BCs[bnd[i,-1]][0]
            uD[topoDim*bnd[i,:-1]+1] = BCs[bnd[i,-1]][1]
            continue
        pts = bzCP[bnd[i,:-1],:topoDim]
        wt = bzCP[bnd[i,:-1],-1]
        spl = BSpline(bzu, pts, degree)
        K = np.zeros((degree+1, degree+1)).astype(float)
        #K_Y = np.zeros((degree+1, degree+1))
        F_X = np.zeros(degree+1).astype(float)
        F_Y = np.zeros(degree+1).astype(float)
        for j in range(len(w)):
            N = pyBernstein(degree, x[j])
            dN = pyDerBernstein(degree, x[j])
            J0 = np.linalg.norm(pts.transpose() @ dN)
            #J_X = np.dot(pts[:,0], dN)
            #J_Y = np.dot(pts[:,1], dN)
            quadXY = spl(x[j])
            K += np.outer(N, N)*J0*w[j]
            F_X += N*J0*w[j]*lagFunc[BCs[bnd[i,-1]]](quadXY)[0]
            F_Y += N*J0*w[j]*lagFunc[BCs[bnd[i,-1]]](quadXY)[1]
        uD[topoDim*bnd[i,:-1]] = np.linalg.solve(K, F_X)
        uD[topoDim*bnd[i,:-1]+1] = np.linalg.solve(K, F_Y)

    return uD  # still building, maybe no need to finish

def queryBCsFromLagrangian2D(bnd, bzCP, BCs, topoDim, degree, lagFunc):
    flavor = 'LeastSquare' # LeastSquare or LinearSystem
    if flavor == 'LeastSquare':
        uD = queryBCs2DFlavorLeastSquare(bnd, bzCP, BCs, topoDim, degree, lagFunc)
    elif flavor == 'LinearSystem':
        uD = queryBCs2DFlavorLinearSystem(bnd, bzCP, BCs, topoDim, degree, lagFunc)
    return uD

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

    #B = np.empty((eldof, topoDim))
    #BT = np.empty((eldof, topoDim))
    #trB = np.zeros((eldof, topoDim))
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
            val[ntriplets:ntriplets+eldof**2] += eleK.flatten('F')
        ntriplets += eldof**2
    K = sparse.coo_matrix((val, (row, col)))
    K = sparse.csr_matrix(K)
    del val, row, col
    F = np.zeros(topoDim*len(bzCP))

    #applyBzDirichletBC(K, F, bnd, problem['BCs'], Var)
    uD = queryBCsFromLagrangian2D(bnd, bzCP, problem['BCs'], topoDim, d, Var[sys_name])
    uDdof = np.concatenate((2*meshData[sys_name]['bzMesh']['uniqueBnd'], 2*meshData[sys_name]['bzMesh']['uniqueBnd']+1))
    
    
    
    #bndNodes = np.unique(2*bnd[:,:-1].flatten())
    #bndNodes = np.concatenate((bndNodes, bndNodes+1))
    #uD = np.zeros(len(bndNodes))
    #uD[np.array([0,1,2,3])] = 0.1
   
    F = F - (K @ uD)
    F[uDdof] = uD[uDdof]
    K[:,uDdof] = 0.0
    K[uDdof,:] = 0.0
    K[uDdof, uDdof] = 1.  # maybe should do something to keep condition numbers
    disp = spsolve(K, F)
    meshData[sys_name]['bzMesh']['cp'][:,0] += disp[0::2]
    meshData[sys_name]['bzMesh']['cp'][:,1] += disp[1::2]
    IO.savemat('out.mat',meshData[sys_name]['bzMesh'])

    return 0

def updateLagrangianViaBz(meshData, sys_name):
    T = meshData[sys_name]['mapBzOrd']
    I = meshData[sys_name]['mapBzElem']
    bzElem = meshData[sys_name]['bzMesh']['elemNode']
    bzCP = meshData[sys_name]['bzMesh']['cp']
    d = meshData[sys_name]['bzMesh']['degree']
    topoDim = meshData[sys_name]['topoDim']
    mesh =  meshData[sys_name]['mesh']
    new_coord = np.empty((len(mesh.coordinates()),topoDim)).astype(float)
    NN = np.empty((6, len(bzElem[0,:])))
    for i in range(len(T)):
        pts = bzCP[bzElem[I[i]],:topoDim]
        weights = bzCP[bzElem[I[i]],-1]
        libTIGA.shapeFunc2(d, T[i,:], weights, NN)
        new_coord[i,:] = pts.transpose() @ NN[0,:]
    mesh.coordinates()[:] = new_coord
    del new_coord
    mesh.bounding_box_tree().build(mesh) # looks like we should update the bounding tree after changing node coords, if not then interpolation is likely to run into errors
    return 0



