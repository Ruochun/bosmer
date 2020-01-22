from dolfin import *
from rmtursSolver import *
from mpi4py import MPI as pmp

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def sigma(u):
    E = 1.
    nu = .3
    mu = E/(2.*(1.+nu))
    lam = E*nu/((1.+nu)*(1.-2.*nu))
    return 2.*mu*epsilon(u) + lam*tr(epsilon(u))*Identity(len(u))

###### now begining form problems ########

def computeObj(meshData, para, Var):
    dx = meshData['fluid']['dx']
    dX = meshData['fluid']['dX']
    T = Var['fluid']['T']
    (u, _) = split(Var['fluid']['up'])
    M1 = para['objWeight']*(-T)*dX(90) # heat obj
    M2 = inner(grad(u), grad(u))*dx # dissp obj
    M3 = Constant(1.)*dx # volume obj
    return assemble(M1), assemble(M2), assemble(M3)-meshData['fluid']['volCons']

def formProblemNS(meshData, BCs, para, Var, system):
    solver_type = system['NS']['nls']
    W = meshData['fluid']['spaceNS']
    (v, q) = TestFunctions(W)
    w = Var['fluid']['up']#Function(W)
    (u, p) = split(w)
    #dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    nu = para['fluid']['nu']

    h_ugn = meshData['fluid']['hmax']
    h_rgn = meshData['fluid']['hmin']
    tau_sugn3 = h_rgn**2/(4.0*nu)
    tau_supg = 1.0/sqrt(4.0/h_ugn**2 + 1.0/tau_sugn3**2)
    tau_pspg = tau_supg
    momEqn = - div(nu*grad(u)) + grad(u)*u + grad(p)
    F_stab = (tau_supg*inner(grad(v)*u, momEqn) + tau_pspg*inner(grad(q), momEqn))*dx
    F = (
            2.*nu*inner(epsilon(u), epsilon(v))
            #2.*nu*inner(grad(u), grad(v))
            - div(v)*p + q*div(u) + dot(v, dot(u, nabla_grad(u)))
        )*dx
    F = F + F_stab
    J = derivative(F, w)
    if solver_type == "rmturs":
        assem = rmtursAssembler(J, F, BCs['fluid']['NS'])
        problem = rmtursNonlinearProblem(assem)
    elif solver_type == "variational":
        problem = NonlinearVariationalProblem(F, w, BCs['fluid']['NS'], J)
    return problem

def formProblemAdjNS(meshData, BCs, para, Var, system):
    W = meshData['fluid']['spaceNS']
    (v, q) = TestFunctions(W)
    solver_type = "variational"
    if solver_type == "rmturs":
        w = Var['fluid']['up_prime']
        (u, p) = split(w)
    elif solver_type == "variational":
        (u, p) = TrialFunctions(W)
    #dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    nu = para['fluid']['nu']
    (u_, p_) = split(Var['fluid']['up'])
    T_ = Var['fluid']['T']
    T_prime = Var['fluid']['T_prime']

    h_ugn = meshData['fluid']['hmax']
    h_rgn = meshData['fluid']['hmin']
    tau_sugn3 = h_rgn**2/(4.0*nu)
    tau_supg = 1.0/sqrt(4.0/h_ugn**2 + 1.0/tau_sugn3**2)
    tau_pspg = tau_supg
    momEqn1 = - div(nu*grad(v)) + grad(v)*u_ + grad(u_)*v + grad(q)
    momEqn2 = - div(nu*grad(u_)) + grad(u_)*u_ + grad(p_)
    F_stab = (tau_supg*inner(grad(u)*u_, momEqn1) + tau_supg*inner(grad(u)*v, momEqn2) + tau_pspg*inner(grad(p), momEqn1))*dx
    #F_stab = (tau_supg*inner(grad(v)*u_, momEqn2) + tau_pspg*inner(grad(q), momEqn2))*dx
    F = (
            2.*nu*inner(epsilon(u), epsilon(v))
            #2.*nu*inner(grad(u), grad(v))
            + div(v)*p 
            + dot(dot(v, nabla_grad(u_)), u) + dot(dot(u_, nabla_grad(v)), u)
            - q*div(u) + dot(grad(T_), v)*T_prime
            - 2.*inner(grad(u_), grad(v))
        )*dx
    F = F + F_stab

    if solver_type == "rmturs":
        J = derivative(F, w)
        assem = rmtursAssembler(J, F, BCs['fluid']['adjNS'])
        problem = rmtursNonlinearProblem(assem)
    elif solver_type == "variational":
        #problem = NonlinearVariationalProblem(F, w, BCs['fluid']['adjNS'], J)
        a, L = lhs(F), rhs(F)
        problem = LinearVariationalProblem(a, L, Var['fluid']['up_prime'], BCs['fluid']['adjNS'])
    return problem

def formProblemThermal(meshData, BCs, para, Var, system):
    Q = meshData['fluid']['spaceThermal']
    S = TestFunction(Q)
    solver_type = "variational"
    if solver_type == "rmturs":
        T = Var['fluid']['T']
    elif solver_type == "variational":
        T = TrialFunction(Q)
        
    #dT = TrialFunction(Q)
    dx = meshData['fluid']['dx']     
    Pe = para['fluid']['Pe'] 
    nu = para['fluid']['nu'] 
    T_hat = para['fluid']['T_hat']
    h_hat = para['fluid']['h_hat']     
    (u_, p_) = split(Var['fluid']['up'])
    ds = meshData['fluid']['ds']

    h_ugn = meshData['fluid']['hmax']
    h_rgn = meshData['fluid']['hmin']
    #tau_sugn3 = h_rgn**2/(4.0*nu)
    tau_supg = 1.0/sqrt(4.0/h_ugn**2)
    #tau_pspg = tau_supg
    momEqn = -1.0/Pe*div(grad(T)) + dot(u_, grad(T))
    F_stab = (tau_supg*dot(grad(S), u_)*momEqn)*dx
    F = (
            (1./Pe)*dot(grad(T), grad(S))
            + dot(grad(T), u_)*S
        )*dx + (
            T*S*h_hat - S*h_hat*T_hat
        )*ds(0)
    F = F + F_stab

    if solver_type == "rmturs":
        J = derivative(F, T)
        assem = rmtursAssembler(J, F, BCs['fluid']['thermal'])
        problem = rmtursNonlinearProblem(assem)
    elif solver_type == "variational":
        a, L = lhs(F), rhs(F)
        problem = LinearVariationalProblem(a, L, Var['fluid']['T'], BCs['fluid']['thermal'])
    
    return problem

def formProblemAdjThermal(meshData, BCs, para, Var, system):
    Q = meshData['fluid']['spaceThermal']
    S = TestFunction(Q)
    solver_type = "variational"
    if solver_type == "rmturs":
        T = Var['fluid']['T_prime']
    elif solver_type == "variational":
        T = TrialFunction(Q)

    #dT = TrialFunction(Q)
    dx = meshData['fluid']['dx']
    dX = meshData['fluid']['dX']
    ds = meshData['fluid']['ds']
    Pe = para['fluid']['Pe']
    T_hat = para['fluid']['T_hat']
    h_hat = para['fluid']['h_hat']
    (u_, p_) = split(Var['fluid']['up'])

    h_ugn = meshData['fluid']['hmax']
    h_rgn = meshData['fluid']['hmin']
    tau_supg = 1.0/sqrt(4.0/h_ugn**2)
    #momEqn = -1.0/Pe*div(grad(T)) + dot(u_, grad(T))
    #F_stab = (tau_supg*dot(grad(S), u_)*momEqn)*dx
    momEqn = -1.0/Pe*div(grad(S)) + dot(u_, grad(S))
    F_stab = tau_supg*dot(grad(T), u_)*momEqn*dx
    F = (
            (1./Pe)*dot(grad(S), grad(T))
            + dot(grad(S), u_)*T
        )*dx + (
            T*S*h_hat
        )*ds(0) + (
            para['objWeight']*S
        )*dX(90)
    F = F + F_stab

    if solver_type == "rmturs":
        J = derivative(F, T)
        assem = rmtursAssembler(J, F, BCs['fluid']['adjThermal'])
        problem = rmtursNonlinearProblem(assem)
    elif solver_type == "variational":
        a, L = lhs(F), rhs(F)
        problem = LinearVariationalProblem(a, L, Var['fluid']['T_prime'], BCs['fluid']['adjThermal'])
    return problem

def formProblemShapeGradient(meshData, BCs, para, Var, system):
    n = meshData['fluid']['n']
    ds = meshData['fluid']['ds']
    dx = meshData['fluid']['dx']
    (u, _) = split(Var['fluid']['up'])
    (u_prime, _) = split(Var['fluid']['up_prime'])
    T = Var['fluid']['T']
    T_prime = Var['fluid']['T_prime']
    nu = para['fluid']['nu']
    Pe = para['fluid']['Pe']
    (w, p) = TrialFunctions(meshData['fluid']['spaceSG'])
    (v, q) = TestFunctions(meshData['fluid']['spaceSG'])
    g = assemble(Constant(1.0)*dx) - meshData['fluid']['volCons']
    
    a = (inner(grad(w) , grad(v))+inner(w,v))*dx+p*inner(v,n)*ds(0)+q*inner(w,n)* ds(0)
    L = (-2.*nu*inner(epsilon(u), epsilon(u_prime))
         -1./Pe*dot(grad(T), grad(T_prime))
         +inner(grad(u), grad(u))
        )*inner(n, v)*ds(0) - g*q*ds(0)
    problem = LinearVariationalProblem(a, L, Var['fluid']['v'], BCs['fluid']['SG'])

    return problem

def formProblemLinearElasticity(meshData, BCs, para, Var, system):
    dx = meshData['fluid']['dx']
    V = meshData['fluid']['spaceLE']
    if V.mesh().topology().dim() == 2:
        zeros = Constant((0., 0.))
    elif V.mesh().topology().dim() == 3:
        zeros = Constant((0., 0., 0.))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v))*dx
    L = inner(zeros, v)*dx
    problem = LinearVariationalProblem(a, L, Var['fluid']['w'], BCs['fluid']['LE'])
    return problem

##### now begin forming solvers ########

def formSolverNonLinearProblem(problem, para, sys_name):
    nonlinear_solver_type = para[sys_name]['nls']
    if nonlinear_solver_type == "rmturs":
        linear_solver = formLinearSolver(para, sys_name)
        solver = rmtursNewtonSolver(linear_solver)
        solver.parameters["relative_tolerance"] = 1e-8
        solver.parameters["error_on_nonconvergence"] = False
        solver.parameters["maximum_iterations"] = 7
    elif nonlinear_solver_type == "variational":
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['relative_tolerance'] = 1E-9
        solver.parameters['newton_solver']['maximum_iterations'] = 7
        if system['ls'] == "iterative":
            solver.parameters['newton_solver']['linear_solver'] = 'gmres'
            solver.parameters['newton_solver']['preconditioner'] = 'default'
            #solver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
            #solver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-5
            solver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 200
            #solver.parameters['newton_solver']['krylov_solver']['restart'] = 20
            #solver.parameters['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

    return solver

def formSolverLinearProblem(problem, para, sys_name):
    solver = LinearVariationalSolver(problem)
    #PETScOptions.clear()
    #PETScOptions.set("ksp_type", "fgmres")
    #PETScOptions.set("ksp_monitor")
    #solver.set_from_options()
    if para[sys_name]['ls'] = 'iterative':
        solver.parameters["linear_solver"] = 'gmres'
    elif para[sys_name]['ls'] = 'direct':
        solver.parameters["linear_solver"] = 'lu'
    solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-6
    solver.parameters["krylov_solver"]["monitor_convergence"] = True
    solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
    solver.parameters["krylov_solver"]["report"] = True
    solver.parameters["krylov_solver"]["error_on_nonconvergence"] = False
    #for item in solver.parameters["krylov_solver"].items(): print(item)
    return solver

def formLinearSolver(para, sys_name):   # rmturs needs its linear_solver object
    linear_solver = PETScKrylovSolver() 
    PETScOptions.clear()
    linear_solver.parameters["relative_tolerance"] = 1e-9
    linear_solver.parameters['error_on_nonconvergence'] = False
    PETScOptions.set("ksp_monitor")
    if para[sys_name]['ls'] == "iterative":
        PETScOptions.set("ksp_type", "fgmres")
        PETScOptions.set("ksp_gmres_restart", 100)
        PETScOptions.set("ksp_max_it", 200)
        PETScOptions.set("ksp_monitor")
        #PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("preconditioner", "default")
        #PETScOptions.set("nonzero_initial_guess", True)
    linear_solver.set_from_options()
    return linear_solver


