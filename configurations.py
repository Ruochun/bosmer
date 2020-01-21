from dolfin import *
from mshr import *

def readinSystemParameters(para, args):

    sys_list = ['NS', 'adjNS', 'thermal', 'adjThermal', 'SG', 'LE'] # list of all system names
    for sys_name in sys_list:
        para[sys_name] = {} # initialize 

    para['NS']['nls'] = args.nls
    
    para['adjNS']['solver_rtol'] = 1e-8
    para['adjNS']['solver_atol'] = 1e-8

    # config linear solver: unless you wish to set them one by one, they can be configured in a whole-sale
    if args.ls != "none":
        for sys_name in sys_list:
            para[sys_name]['ls'] = args.ls
 
def readinPhysicalParameters(para, args):
    
    return

