from dolfin import *
import generalUtilities as gU
from mshr import *
import numpy

def sampleMesh(res=50):
    domain_r = (
                Rectangle(Point(0.,0.), Point(3.,1.))
                - Rectangle(Point(1.,.4), Point(2.,.6))
               )    
    mesh = generate_mesh(domain_r, res)
    return mesh

def fluidBCs():

    # define your BCs
    # NOT IMPLEMENTED!
    BC['NS']['periodic'] = 'x'
    BC['NS']['essential'] = [(0, Constant(0., 0.))]

    return 0

def markSubDomains(mesh):
    subDomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    subDomains.set_all(99)
    class outflowCV(SubDomain):
        def inside(self, x, on_boundary):
            return not(on_boundary) and (x[0]>2.8) 
    outflowCV().mark(subDomains, 90)
    return subDomains     

def markBoundaries(mesh):
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary.set_all(99)
    class solidWall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    class inflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0]<DOLFIN_EPS
    class outflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0]>3.-DOLFIN_EPS
    class slipWall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[1]<DOLFIN_EPS or x[1]>1.-DOLFIN_EPS)

    solidWall().mark(boundary, 0)
    inflow().mark(boundary, 1)
    outflow().mark(boundary, 2)
    slipWall().mark(boundary, 90)
    return boundary    

def applyNSBCs(meshData, markers):
    W = meshData['fluid']['spaceNS']
    u0 = 1.0
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
        inflow = Expression(('u_in', '0.0'), u_in=u0, degree=2)
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
        inflow = Expression(('u_in', '0.0', '0.0'), u_in=u0, degree=2)
    bc0 = DirichletBC(W.sub(0), noslip, markers, 0)
    bc1 = DirichletBC(W.sub(0), inflow, markers, 1)
    bc90 = DirichletBC(W.sub(0).sub(1), 0.0, markers, 90)
    return [bc0, bc1] # slip not added

def applyAdjNSBCs(meshData, markers):
    W = meshData['fluid']['spaceNS']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc0 = DirichletBC(W.sub(0), noslip, markers, 0)
    bc1 = DirichletBC(W.sub(0), noslip, markers, 1)
    bc90 = DirichletBC(W.sub(0).sub(1), 0.0, markers, 90)
    return [bc0, bc1] # slip not added

def applyThermalBCs(meshData, markers):
    W = meshData['fluid']['spaceThermal']
    bc1 = DirichletBC(W, 0.0, markers, 1)
    return [bc1]

def applyAdjThermalBCs(meshData, markers):
    W = meshData['fluid']['spaceThermal']
    bc1 = DirichletBC(W, 0.0, markers, 1)
    return [bc1]

def applyShapeGradientBCs(meshData, markers):
    W = meshData['fluid']['spaceSG']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc1 = DirichletBC(W.sub(0), noslip, markers, 1)
    bc2 = DirichletBC(W.sub(0), noslip, markers, 2)
    bc90 = DirichletBC(W.sub(0), noslip, markers, 90)
    return [bc1, bc2, bc90]

def applyLinearElasticityBCs(meshData, markers, Var, para):
    alpha = para['stepLen']
    v = Var['fluid']['modified_v']
    W = meshData['fluid']['spaceLE']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc0 = DirichletBC(W, v, markers, 0)
    bc1 = DirichletBC(W, noslip, markers, 1)
    bc2 = DirichletBC(W, noslip, markers, 2)    
    bc90 = DirichletBC(W, noslip, markers, 90)
    return [bc0, bc1, bc2, bc90]


