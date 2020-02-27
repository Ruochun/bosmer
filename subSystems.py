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

def flowDirection(u):
    if len(u) == 2:
        D = Expression(("1.0","0.0"), degree=1)
    elif len(u) == 3:
        D = Expression(("1.0","0.0","0.0"), degree=1) 
    return D
###### now begining form problems ########

def computeObj(meshData, para, Var):
    dx = meshData['fluid']['dx']
    dX = meshData['fluid']['dX']
    T = Var['fluid']['T']
    (u, _) = split(Var['fluid']['up'])
    flowDir = flowDirection(u)
    M1 = para['objWeight']*(-T)*dot(u, flowDir)*dX(90) # heat obj
    M_T = T*dX(90)
    #M1 = para['objWeight']*(-T)*dX(90)
    M2 = inner(grad(u), grad(u))*dx # dissp obj
    M3 = Constant(1.)*dx # volume obj
    return assemble(M1), assemble(M2), assemble(M3)-meshData['fluid']['volCons'], assemble(M_T)/assemble(Constant(1.)*dX(90))

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
    #F = F + F_stab
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
    solver_type = system['adjNS']['nls']
    if solver_type == "rmturs":
        w = Var['fluid']['up_prime']
        (u, p) = split(w)
    elif solver_type == "variational":
        (u, p) = TrialFunctions(W)
    #dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    dX = meshData['fluid']['dX']
    nu = para['fluid']['nu']
    (u_, p_) = split(Var['fluid']['up'])
    T_ = Var['fluid']['T']
    T_prime = Var['fluid']['T_prime']

    h_ugn = meshData['fluid']['hmax']
    h_rgn = meshData['fluid']['hmin']
    flowDir = flowDirection(u_)
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
        )*dx + (
            para['objWeight']*dot(v, flowDir)*T_  # a term originated from including flow dir in the obj
        )*dX(90)
    #F = F + F_stab

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
            0.#T*S*h_hat - S*h_hat*T_hat
        )*ds(0)
    #F = F + F_stab

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
    flowDir = flowDirection(u_)
    tau_supg = 1.0/sqrt(4.0/h_ugn**2)
    #momEqn = -1.0/Pe*div(grad(T)) + dot(u_, grad(T))
    #F_stab = (tau_supg*dot(grad(S), u_)*momEqn)*dx
    momEqn = -1.0/Pe*div(grad(S)) + dot(u_, grad(S))
    F_stab = tau_supg*dot(grad(T), u_)*momEqn*dx
    F = (
            (1./Pe)*dot(grad(S), grad(T))
            + dot(grad(S), u_)*T
        )*dx + (
            0.#T*S*h_hat
        )*ds(0) + (
            para['objWeight']*S*dot(u_, flowDir)
        )*dX(90)
    #F = F + F_stab

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
        for option in para[sys_name]['general']:
            solver.parameters[option] = para[sys_name]['general'][option]

    # FEniCS variational nonlinear solver for debug purpose
    elif nonlinear_solver_type == "variational":
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['relative_tolerance'] = 1E-8
        solver.parameters['newton_solver']['maximum_iterations'] = 7
        if para[sys_name]['ls'] == "iterative":
            solver.parameters['newton_solver']['linear_solver'] = 'gmres'
            solver.parameters['newton_solver']['preconditioner'] = 'default'
            #solver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
            solver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 3E-5
            solver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 100
            #solver.parameters['newton_solver']['krylov_solver']['restart'] = 50
            #solver.parameters['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
    return solver

def formSolverLinearProblem(problem, para, sys_name):
    solver = LinearVariationalSolver(problem)
    """
    for component in para[sys_name]: # components typically include a 'general'(high-level) block and a 'krylov_solver'(linear solver specs) block
        if not(isinstance(para[sys_name][component], dict)):
            continue

        if component == "general":
            for option in para[sys_name]["general"]:
                solver.parameters[option] = para[sys_name]["general"][option]
        else:
            for option in para[sys_name][component]:
                solver.parameters[component][option] = para[sys_name][component][option]
    """
    #for item in solver.parameters["krylov_solver"].items(): print(item)
    return solver

def formLinearSolver(para, sys_name):   # rmturs needs its linear_solver objecti
    # these linear options are exposed in FEniCS, haven't found a way to set them in PETScOptions
    exposed_list = ['relative_tolerance', 'absolute_tolerance', 'error_on_nonconvergence']

    linear_solver = PETScKrylovSolver() 
    PETScOptions.clear()
    for option in para[sys_name]['krylov_solver']:
        if option in exposed_list:
            linear_solver.parameters[option] = para[sys_name]['krylov_solver'][option]
        else:
            if para[sys_name]['krylov_solver'][option] != []:
                PETScOptions.set(option, para[sys_name]['krylov_solver'][option])
            else:
                PETScOptions.set(option)

    linear_solver.set_from_options()
    return linear_solver


