from dolfin import *
from rmtursSolver import *

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def formProblemNS(meshData, BCs, para, Var, system):
    W = meshData['fluid']['spaceNS']
    (v, q) = TestFunctions(W)
    w = Var['fluid']['up']#Function(W)
    (u, p) = split(w)
    #dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    nu = para['fluid']['nu']

    F = (
            2.*nu*inner(epsilon(u), epsilon(v))
            - div(v)*p + q*div(u) + dot(v, dot(u, nabla_grad(u)))
        )*dx
    J = derivative(F, w)
    if system['ns'] == "rmturs":
        assem = rmtursAssembler(J, F, BCs['fluid']['NS'])
        problem = rmtursNonlinearProblem(assem)
    elif system['ns'] == "variational":
        problem = NonlinearVariationalProblem(F, w, BCs['fluid']['NS'], J)
    return problem

def formProblemAdjNS(meshData, BCs, para, Var, system):
    W = meshData['fluid']['spaceNS']
    (v, q) = TestFunctions(W)
    w = Var['fluid']['up_prime']
    (u, p) = split(w)
    #dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    nu = para['fluid']['nu']
    (u_, p_) = split(Var['fluid']['up'])
    T_ = Var['fluid']['T']
    T_prime = Var['fluid']['T_prime']

    F = (
            2.*nu*inner(epsilon(u), epsilon(v))
            - div(v)*p 
            + dot(dot(v, nabla_grad(u_)), u) + dot(dot(u_, nabla_grad(v)), u)
            + q*div(u) + dot(grad(T_), v)*T_prime
        )*dx
    J = derivative(F, w)
    if system['ns'] == "rmturs":
        assem = rmtursAssembler(J, F, BCs['fluid']['adjNS'])
        problem = rmtursNonlinearProblem(assem)
    elif system['ns'] == "variational":
        problem = NonlinearVariationalProblem(F, w, BCs['fluid']['adjNS'], J)
    
    return problem

def formProblemThermal(meshData, BCs, para, Var, system):
    Q = meshData['fluid']['spaceThermal']
    S = TestFunction(Q)
    if system['ns'] == "rmturs":
        T = Var['fluid']['T']
    elif system['ns'] == "variational":
        T = TrialFunction(Q)
        
    #dT = TrialFunction(Q)
    dx = meshData['fluid']['dx']
    Pe = para['fluid']['Pe']
    (u_, p_) = split(Var['fluid']['up'])
    ds = meshData['fluid']['ds']
    F = (
            (1./Pe)*dot(grad(T), grad(S))
            + dot(grad(T), u_)*S
        )*dx + (0.
            
        )*ds(0)
    
    if system['ns'] == "rmturs":
        J = derivative(F, T)
        assem = rmtursAssembler(J, F, BCs['fluid']['thermal'])
        problem = rmtursNonlinearProblem(assem)
    elif system['ns'] == "variational":
        a, L = lhs(F), rhs(F)
        problem = LinearVariationalProblem(a, L, Var['fluid']['T'], BCs['fluid']['thermal'])
    
    return problem

def formProblemAdjThermal(meshData, BCs, para, Var, system):
    Q = meshData['fluid']['spaceThermal']
    S = TestFunction(Q)
    if system['ns'] == "rmturs":
        T = Var['fluid']['T_prime']
    elif system['ns'] == "variational":
        T = TrialFunction(Q)

    #dT = TrialFunction(Q)
    dx = meshData['fluid']['dx']
    ds = meshData['fluid']['ds']
    Pe = para['fluid']['Pe']
    (u_, p_) = split(Var['fluid']['up'])

    F = (
            (1./Pe)*dot(grad(S), grad(T))
            + dot(grad(S), u_)*T
        )*dx + (0.
            
        )*ds(0)

    if system['ns'] == "rmturs":
        J = derivative(F, T)
        assem = rmtursAssembler(J, F, BCs['fluid']['adjThermal'])
        problem = rmtursNonlinearProblem(assem)
    elif system['ns'] == "variational":
        a, L = lhs(F), rhs(F)
        problem = LinearVariationalProblem(a, L, Var['fluid']['T_prime'], BCs['fluid']['adjThermal'])
    return problem

def formShapeGradient(meshData):
    n = meshData['fluid']['n']
    ds = meshData['fluid']['ds']

#def searchDirection(meshData, Var):
    
    

def formSolverNS(problem, system):
    if system['ns'] == "rmturs":
        linear_solver = formLinearSolver('generic', system)
        solver = rmtursNewtonSolver(linear_solver)
        solver.parameters["relative_tolerance"] = 1e-4
        solver.parameters["error_on_nonconvergence"] = False
        solver.parameters["maximum_iterations"] = 5
    elif system['ns'] == "variational":
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['relative_tolerance'] = 1E-4
        solver.parameters['newton_solver']['maximum_iterations'] = 5

    return solver

def formSolverThermal(problem, system):
    if system['ns'] == "rmturs":
        linear_solver = formLinearSolver('generic', system)
        solver = rmtursNewtonSolver(linear_solver)
    elif system['ns'] == "variational":
        solver = LinearVariationalSolver(problem)

    return solver

def formLinearSolver(instr, system):   # rmturs needs its linear_solver object
    if instr == "generic":
        linear_solver = PETScKrylovSolver() 
        PETScOptions.clear()
        linear_solver.parameters["relative_tolerance"] = 1e-5
        linear_solver.parameters['error_on_nonconvergence'] = False
        PETScOptions.set("ksp_monitor")
        if system['ls'] == "iterative":
            PETScOptions.set("ksp_type", "fgmres")
            PETScOptions.set("ksp_gmres_restart", 10)
            PETScOptions.set("ksp_max_it", 100)
            PETScOptions.set("pc_type", "hypre")
            #PETScOptions.set("preconditioner", "default")
            #PETScOptions.set("nonzero_initial_guess", True)
        linear_solver.set_from_options()
    return linear_solver


