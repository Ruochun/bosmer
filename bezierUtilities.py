from dolfin import *
import scipy.interpolate as SPI
import scipy.spatial as SPS
import scipy.io as IO
import numpy as np
from ctypes import *

libTIGA = CDLL("./c_lib/lib_tIGA.so")
libTIGA.shapeFunc2.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64), np.ctypeslib.ndpointer(dtype=np.float64)]

def loadMatlabBezierMesh(args):
    mesh = IO.loadmat(args.tiga_mesh_file)
    return mesh

def formMapBezier2Lagrangian(bzMesh, lagMesh, topoDim):


