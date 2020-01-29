from dolfin import *
from mshr import *

def readinSystemParameters(para, args):

    readinEssentials(para, args)

    # the following is about the solvers
    sys_list = ['NS', 'adjNS', 'thermal', 'adjThermal', 'SG', 'LE'] # list of all system names
    component_list = ['general', 'krylov_solver']
    for sys_name in sys_list:
        para[sys_name] = {} # initialize
        for cpn_name in component_list:
            para[sys_name][cpn_name] = {}

    # config linear solver: 'direct' is for debugging; 'ls' is a choice between 'direct' and 'iterative', not the actual iterative solver name (e.g. gmres)
    if args.ls == 'direct':
        for sys_name in sys_list:
            para[sys_name]['ls'] = 'direct'
            para[sys_name]['general']['linear_solver'] = 'mumps'
    elif args.ls == 'iterative':
        for sys_name in sys_list:
            para[sys_name]['ls'] = 'iterative'
            para[sys_name]['general']['linear_solver'] = 'gmres'
   
    # for NS solver 
    para['NS']['nls'] = args.nls
    para['NS']['general']['relative_tolerance'] = 1e-10
    para['NS']['general']['error_on_nonconvergence'] = False
    para['NS']['general']['maximum_iterations'] = 7
    para['NS']['krylov_solver']['ksp_type'] = 'fgmres'
    para['NS']['krylov_solver']["relative_tolerance"] = 1e-8
    para['NS']['krylov_solver']['error_on_nonconvergence'] = False
    para['NS']['krylov_solver']['ksp_gmres_restart'] = 300
    para['NS']['krylov_solver']['ksp_max_it'] = 500
    #para['NS']['krylov_solver']['ksp_monitor'] = []
    para['NS']['krylov_solver']['preconditioner'] = 'default'

    # for advective thermal solver
    para['thermal']['general']['preconditioner'] = 'hypre_amg'
    para['thermal']['krylov_solver']['relative_tolerance'] = 1e-8
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
    para['adjThermal']['krylov_solver']['relative_tolerance'] = 1e-8
    #para['adjThermal']['krylov_solver']['monitor_convergence'] = True
    para['adjThermal']['krylov_solver']['maximum_iterations'] = 300
    para['adjThermal']['krylov_solver']['report'] = True
    para['adjThermal']['krylov_solver']['error_on_nonconvergence'] = False

    # for adjoint NS solver    
    para['adjNS']['general']['preconditioner'] = 'sor'
    para['adjNS']['krylov_solver']['relative_tolerance'] = 1e-6
    para['adjNS']['krylov_solver']['monitor_convergence'] = True
    para['adjNS']['krylov_solver']['maximum_iterations'] = 2500
    para['adjNS']['krylov_solver']['report'] = True
    para['adjNS']['krylov_solver']['error_on_nonconvergence'] = False

    # for shape gradient calculation
    para['SG']['general']['preconditioner'] = 'default'
    para['SG']['krylov_solver']['relative_tolerance'] = 1e-8
    #para['SG']['krylov_solver']['monitor_convergence'] = True
    para['SG']['krylov_solver']['maximum_iterations'] = 300
    para['SG']['krylov_solver']['report'] = True
    para['SG']['krylov_solver']['error_on_nonconvergence'] = False

    # for linear elasticity mesh motion 
    para['LE']['general']['preconditioner'] = 'default'
    para['LE']['krylov_solver']['relative_tolerance'] = 1e-8
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

    return


