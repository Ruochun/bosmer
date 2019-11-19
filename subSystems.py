from fenics import *

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def formProblemNS(meshData, W, BCs, para):
    (v, q) = TestFunctions(W)
    w = Function(W)
    (u, p) = split(w)
    dw = TrialFunction(W)
    
    F = (
            2.*nu*inner(epsilon(
