from dolfin import *
from mshr import *

def readinSystemParameters(para, args):
    para['fluid']['adjNS']['solver_rtol'] = 1e-8
    para['fluid']['adjNS']['solver_atol'] = 1e-8

def readinPhysicsParameters(para, args):
    
    return

