from dolfin import *
from mshr import *

def readinSystemParameters(para, args):

    sys_list = ['NS', 'adjNS', 'thermal', 'adjThermal', 'SG', 'LE'] # list of all system names
    for sys_name in sys_list:
        para[sys_name] = {} # initialize 

    para['NS']['nls'] = args.nls

    para['thermal']['linear_solver'] = 'gmres'
    para['thermal']['krylov_solver']['relative_tolerance'] = 1e-6
    para['thermal']['krylov_solver']['monitor_convergence'] = True
    para['thermal']['krylov_solver']['maximum_iterations'] = 300
    para['thermal']['krylov_solver']['report'] = True
    para['thermal']['krylov_solver']['error_on_nonconvergence'] = False

    para['adjNS']['solver_rtol'] = 1e-8
    para['adjNS']['solver_atol'] = 1e-8

    # config linear solver: 'direct' is for debugging. 'ls' can be configured in a whole-sale
    for sys_name in sys_list:
        para[sys_name]['ls'] = args.ls

def readinPhysicalParameters(para, args):
    
    return

