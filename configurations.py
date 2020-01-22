from dolfin import *
from mshr import *

def readinSystemParameters(para, args):

    sys_list = ['NS', 'adjNS', 'thermal', 'adjThermal', 'SG', 'LE'] # list of all system names
    for sys_name in sys_list:
        para[sys_name] = {} # initialize
        para[sys_name]['krylov_solver'] = {}
        para[sys_name]['PETScOptions'] = {} 

    para['NS']['nls'] = args.nls

    # for advective thermal solver
    para['thermal']['linear_solver'] = 'gmres'
    para['thermal']['preconditioner'] = 'hypre_amg'
    para['thermal']['krylov_solver']['relative_tolerance'] = 1e-6
    para['thermal']['krylov_solver']['monitor_convergence'] = True
    para['thermal']['krylov_solver']['maximum_iterations'] = 300
    para['thermal']['krylov_solver']['report'] = True
    para['thermal']['krylov_solver']['error_on_nonconvergence'] = False
    #para['thermal']['PETScOptions']['pc_type'] = 'hypre'
    #para['thermal']['PETScOptions']['pc_hypre_type'] = 'boomeramg'
    #para['thermal']['PETScOptions']['pc_hypre_boomeramg_coarsen_type'] = 'hmis'
    #para['thermal']['PETScOptions']['pc_hypre_boomeramg_interp_type'] = 'ext+i'
    #para['thermal']['PETScOptions']['pc_hypre_boomeramg_p_max'] = 4
    #para['thermal']['PETScOptions']['boomeramg_agg_nl'] = 1

    # for adjoint advective thermal solver
    para['adjThermal']['linear_solver'] = 'gmres'
    para['adjThermal']['preconditioner'] = 'hypre_amg'
    para['adjThermal']['krylov_solver']['relative_tolerance'] = 1e-6
    para['adjThermal']['krylov_solver']['monitor_convergence'] = True
    para['adjThermal']['krylov_solver']['maximum_iterations'] = 300
    para['adjThermal']['krylov_solver']['report'] = True
    para['adjThermal']['krylov_solver']['error_on_nonconvergence'] = False

    # for adjoint NS solver    
    para['adjNS']['linear_solver'] = 'gmres'
    para['adjNS']['preconditioner'] = 'default'
    para['adjNS']['krylov_solver']['relative_tolerance'] = 1e-6
    para['adjNS']['krylov_solver']['monitor_convergence'] = True
    para['adjNS']['krylov_solver']['maximum_iterations'] = 1000
    para['adjNS']['krylov_solver']['report'] = True
    para['adjNS']['krylov_solver']['error_on_nonconvergence'] = False

    # for shape gradient calculation
    para['SG']['linear_solver'] = 'gmres'
    para['SG']['preconditioner'] = 'default'
    para['SG']['krylov_solver']['relative_tolerance'] = 1e-6
    para['SG']['krylov_solver']['monitor_convergence'] = True
    para['SG']['krylov_solver']['maximum_iterations'] = 300
    para['SG']['krylov_solver']['report'] = True
    para['SG']['krylov_solver']['error_on_nonconvergence'] = False

    # for linear elasticity mesh motion 
    para['LE']['linear_solver'] = 'gmres'
    para['LE']['preconditioner'] = 'default'
    para['LE']['krylov_solver']['relative_tolerance'] = 1e-6
    para['LE']['krylov_solver']['monitor_convergence'] = True
    para['LE']['krylov_solver']['maximum_iterations'] = 300
    para['LE']['krylov_solver']['report'] = True
    para['LE']['krylov_solver']['error_on_nonconvergence'] = False

    # config linear solver: 'direct' is for debugging; 'ls' is a choice between 'direct' and 'iterative', not the actual iterative solver name (e.g. gmres)
    for sys_name in sys_list:
        para[sys_name]['ls'] = args.ls

def readinPhysicalParameters(para, args):
    
    return

