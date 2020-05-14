from dolfin import *
from mshr import *
import numpy as np

def readinSystemParameters(para, args):

    readinEssentials(para, args)

    # readin all default paras; including all the possible systems and the associated components that needs to be configured
    sys_list, component_list = readinDefaults(para)

    # config nonlinear solver: rmturs is generally for nonlinear problems; variational should be used for debugging
    if args.nls == "rmturs":
        for sys_name in sys_list:
            para[sys_name]['nls'] = "rmturs"

    # config linear solver: 'direct' is for debugging; 'ls' is a choice between 'direct' and 'iterative', not the actual iterative solver name (e.g. gmres)
    if args.ls == 'iterative':
        for sys_name in sys_list:
            para[sys_name]['ls'] = "iterative"
            para[sys_name]['general']['linear_solver'] = "gmres"
   
    # for NS solver 
    para['NS']['general']['relative_tolerance'] = 1e-10
    para['NS']['general']['error_on_nonconvergence'] = False
    para['NS']['general']['maximum_iterations'] = 10
    para['NS']['krylov_solver']['ksp_type'] = 'fgmres'
    para['NS']['krylov_solver']["relative_tolerance"] = 1e-8
    para['NS']['krylov_solver']['error_on_nonconvergence'] = False
    #para['NS']['krylov_solver']['ksp_gmres_restart'] = 500
    para['NS']['krylov_solver']['ksp_max_it'] = 250
    #para['NS']['krylov_solver']['ksp_monitor'] = []
    para['NS']['krylov_solver']['preconditioner'] = 'default'

    # for adjoint NS solver
    para['adjNS']['general']['relative_tolerance'] = 1e-8
    para['adjNS']['general']['error_on_nonconvergence'] = False
    para['adjNS']['general']['maximum_iterations'] = 1
    para['adjNS']['krylov_solver']['ksp_type'] = 'gmres'
    para['adjNS']['krylov_solver']["relative_tolerance"] = 1e-9
    para['adjNS']['krylov_solver']['error_on_nonconvergence'] = False
    para['adjNS']['krylov_solver']['ksp_max_it'] = 2000
    para['adjNS']['krylov_solver']['ksp_gmres_restart'] = 500
    para['adjNS']['krylov_solver']['ksp_monitor'] = [] 
    para['adjNS']['krylov_solver']['preconditioner'] = 'default'
    """
    para['adjNS']['krylov_solver']['pc_type'] = 'hypre'
    para['adjNS']['krylov_solver']['pc_hypre_type'] = 'boomeramg'
    para['adjNS']['krylov_solver']['pc_hypre_boomeramg_stong_threshold'] = .7
    para['adjNS']['krylov_solver']['pc_hypre_boomeramg_coarsen_type'] = "hmis"
    para['adjNS']['krylov_solver']["pc_hypre_boomeramg_interp_type"] = "ext+i"
    #para['adjNS']['krylov_solver']["pc_hypre_boomeramg_p_max"] = 4
    #para['adjNS']['krylov_solver']["boomeramg_agg_nl"] = 1
    para['adjNS']['krylov_solver']["nonzero_initial_guess"] = True
    """

    # for advective thermal solver
    para['thermal']['general']['preconditioner'] = 'hypre_amg'
    para['thermal']['krylov_solver']['relative_tolerance'] = 1e-12
    #para['thermal']['krylov_solver']['monitor_convergence'] = True
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
    para['adjThermal']['general']['preconditioner'] = 'hypre_amg'
    para['adjThermal']['krylov_solver']['relative_tolerance'] = 1e-12
    #para['adjThermal']['krylov_solver']['monitor_convergence'] = True
    para['adjThermal']['krylov_solver']['maximum_iterations'] = 300
    para['adjThermal']['krylov_solver']['report'] = True
    para['adjThermal']['krylov_solver']['error_on_nonconvergence'] = False

    # for shape gradient calculation
    para['SG']['general']['preconditioner'] = 'default'
    para['SG']['krylov_solver']['relative_tolerance'] = 1e-12
    #para['SG']['krylov_solver']['monitor_convergence'] = True
    para['SG']['krylov_solver']['maximum_iterations'] = 300
    para['SG']['krylov_solver']['report'] = True
    para['SG']['krylov_solver']['error_on_nonconvergence'] = False

    # for linear elasticity mesh motion 
    para['LE']['general']['preconditioner'] = 'default'
    para['LE']['krylov_solver']['relative_tolerance'] = 1e-12
    #para['LE']['krylov_solver']['monitor_convergence'] = True
    para['LE']['krylov_solver']['maximum_iterations'] = 300
    para['LE']['krylov_solver']['report'] = True
    para['LE']['krylov_solver']['error_on_nonconvergence'] = False
    
    return

def readinPhysicalParameters(para, args):
    
    para['fluid']['nu'] = args.viscosity
    para['fluid']['Pe'] = args.Pe
    para['stepLen'] = args.step_length
    para['fluid']['T_hat'] = 373.
    para['fluid']['h_hat'] = .1
    para['objWeight'] = args.obj_weight
    
    return

def readinEssentials(para, args):

    if args.ts_per_rm>0:
        para['ts_per_rm'] = args.ts_per_rm
    else:
        para['ts_per_rm'] = 9223372036854775800
    para['maxIter'] = args.max_iter
    para['ts_per_out'] = args.ts_per_out
    para['fluidMesh']['recRes'] = args.recRes
    para['fluidMesh']['stiffening_scale'] = args.stiffening_scale
    para['no_stab'] = args.no_stab

    return

def defineProblemTIGALE(meshData, sys_name):
    if meshData[sys_name]['topoDim']==2:
        zeroVec = (0.0,0.0)
    elif meshData[sys_name]['topoDim']==3:
        zeroVec = (0.0,0.0,0.0)
    bc0 = {1:zeroVec}
    bc1 = {0:"modified_v"}
    BCs = {**bc0, **bc1}

    E = 1e3
    nu = 0.3
    D = E/(1.0-nu**2)*np.array([[1.0,nu,0],[nu,1.0,0.0],[0.0,0.0,0.5*(1.0-nu)]])
    return {"D":D, "BCs": BCs}

def readinDefaults(para):
    sys_list = ['NS', 'adjNS', 'thermal', 'adjThermal', 'SG', 'LE'] # list of all system names
    component_list = ['general', 'krylov_solver']
    for sys_name in sys_list:
        para[sys_name] = {} # initialize
        para[sys_name]['nls'] = "variational"
        para[sys_name]['ls'] = "direct"
        for cpn_name in component_list:
            para[sys_name][cpn_name] = {}
            para[sys_name]['general']['linear_solver'] = 'mumps'
    return sys_list, component_list


