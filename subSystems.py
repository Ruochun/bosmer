from dolfin import *
from rmtursSolver import *

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def formProblemNS(meshData, W, BCs, para, Var):
    (v, q) = TestFunctions(W)
    w = Function(W)
    (u, p) = split(w)
    dw = TrialFunction(W)
    dx = meshData['fluid']['dx']
    nu = para['fluid']['nu']
    
    F = (
            2.*nu*inner(epsilon(u), epsilon(v))
            - div(v)*p + q*div(u) + dot(v, dot(u, nabla_grad(u)))
        )*dx
    J = derivative(F, w, dw)
    assem = rmtursAssembler(J, F, BCs['fluid']['NS'])
    problem = rmtursNonlinearProblem(assem)
    return problem

def formProblemAdjNS(meshData, W, BCs, para, Var):
    (v, q) = TestFunctions(W)
    w = Function(W)
    (u, p) = split(w)
    dw = TrialFunction(W)
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
    J = derivative(F, w, dw)
    assem = rmtursAssembler(J, F, BCs['fluid']['adjNS'])
    problem = rmtursNonlinearProblem(assem)
    return problem

def formProblemThermal(meshData, Q, BCs, para, Var):
    S = TestFunction(Q)
    T = Function(Q)
    dT = TrialFunction(Q)
    dx = meshData['fluid']['dx']
    Pe = para['fluid']['Pe']
    (u_, p_) = split(Var['fluid']['up'])
    ds = meshData['fluid']['ds']
    F = (
            (1./Pe)*dot(grad(T), grad(S))
            + dot(grad(T), u_)*S
        )*dx + (
            
        )*ds(0)
    
    J = derivative(F, T, dT)
    assem = rmtursAssembler(J, F, BCs['fluid']['thermal'])
    problem = rmtursNonlinearProblem(assem)
    return problem

def formProblemAdjThermal(meshData, Q, BCs, para, Var):
    S = TestFunction(Q)
    T = Function(Q)
    dT = TrialFunction(Q)
    
    dx = meshData['fluid']['dx']
    ds = meshData['fluid']['ds']
    Pe = para['fluid']['Pe']
    (u_, p_) = split(Var['fluid']['up'])

    F = (
            (1./Pe)*dot(grad(S), grad(T))
            + dot(grad(S), u_)*T
        )*dx + (
            
        )*ds(0)

    J = derivative(F, T, dT)
    assem = rmtursAssembler(J, F, BCs['fluid']['adjThermal'])
    problem = rmtursNonlinearProblem(assem)
    return problem


