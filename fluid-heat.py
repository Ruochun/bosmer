from dolfin import *
#from matplotlib import pyplot
from rmtursSolver import *

import numpy as np
import meshUtilities as mU
import generalUtilities as gU
import subSystems as sS

from mpi4py import MPI as pmp
import argparse, sys, os, gc
import time

commmpi = pmp.COMM_WORLD
# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ns", type=str, dest="ns", default="variational",
                    choices=["variational", "rmturs"],help="non-linear solver, rmturs or FEniCS variational solver")
parser.add_argument("-refine_level","-l", type=int, dest="level", default=0,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.2,
                    help="kinematic viscosity")
parser.add_argument("--Pe", type=float, dest="Pe", default=10.,
                    help="Peclet number for thermal system")
parser.add_argument("--obj_weight", "-w", type=float, dest="obj_weight", default=1.,
                    help="Add a multiplier (weight) to the temperature objective, emphasis it more")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solver, choose from direct or iterative")
parser.add_argument("--ts_per_out", type=int, dest="ts_per_out", default=1,
                    help="number of time steps per output file")
parser.add_argument("--ts_per_rm", type=int, dest="ts_per_rm", default=5,
                    help="number of time steps per remesh, use -1 to never remesh")
parser.add_argument("--mesh_file", "-m", type=str, dest="mesh_file", default="__FINS",
                    help="path and file name of the mesh, or add \'__\' in front of the mesh to use one of the default sample meshes")
parser.add_argument("--out_folder", type=str, dest="out_folder", default="./result",
                    help="output folder name")
parser.add_argument("--max_iter", type=int, dest="max_iter", default=10,
                    help="total number of iteration steps")
parser.add_argument("--sl", "-s", type=float, dest="step_length", default=.005,
                    help="optimization step length multiplier")
parser.add_argument("--recommend_resolution", "-r", type=int, dest="recRes", default=-1,
                    help="instruct the code to generate mesh that has at least this many elements")
parser.add_argument("--volume_constraint", "-v", type=str, dest="volCons", default="1*",
                    help="volume constraint, supply a number, or a number and a '\*'\ in the end allowing the initial domain to expand that many times larger (or smaller)")
args = parser.parse_args(sys.argv[1:])

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["std_out_all_processes"] = False
rank = commmpi.Get_rank()
root = 0
if rank == 0:
    set_log_level(20)
else:
    set_log_level(50)

# prepare data dictionaries
meshData = {} # mesh and related info
BCs = {} # BC sets
funcVar = {} # state and adjoint variables
physicalPara = {} # physical parameters of this problem
systemPara = {} # system parameters
outputData = {} # outputs for postprocessing

# we have a fluid system
meshData['fluid'] = {}
BCs['fluid'] = {}
funcVar['fluid'] = {}
physicalPara['fluid'] = {}
systemPara['fluid'] = {}
outputData['objHeat'] = []
outputData['objDissp'] = [] 
# we don't currently have a solid system


# load in system settings and physical parameters
physicalPara['fluid']['nu'] = args.viscosity
physicalPara['fluid']['Pe'] = args.Pe
physicalPara['stepLen'] = args.step_length
physicalPara['fluid']['T_hat'] = 100.
physicalPara['fluid']['h_hat'] = .1
physicalPara['objWeight'] = args.obj_weight

systemPara['ns'] = args.ns
systemPara['ls'] = args.ls
if args.ts_per_rm>0:
    systemPara['ts_per_rm'] = args.ts_per_rm
else:
    systemPara['ts_per_rm'] = 9223372036854775807
systemPara['maxIter'] = args.max_iter
systemPara['ts_per_out'] = args.ts_per_out
systemPara['fluid']['recRes'] = args.recRes

if args.mesh_file[:2] == "__":
    meshData['fluid']['mesh'], meshData['fluid']['bndExPts'] = mU.sampleMesh(systemPara, args.mesh_file[2:])
    mesh = meshData['fluid']['mesh']
else:
    try:
        meshData['fluid']['mesh'] = Mesh(args.mesh_file)
        mesh = meshData['fluid']['mesh']
    except:
        try:
            meshData['fluid']['mesh'] = Mesh()
            mesh = meshData['fluid']['mesh']
            fid = HDF5File(commmpi, args.mesh_file, 'r')
            fid.read(mesh, 'mesh', False)
            fid.close()
        except:
            info("This mesh is not valid to read in.")
    meshData['fluid']['bndExPts'] = []

    for i in range(args.level):
        mesh = refine(mesh)

assert mesh is not None

if args.volCons[-1] == "*":
    meshData['fluid']['initVol'] = float(args.volCons[:-1])*assemble(Constant(1.)*Measure("dx", domain=mesh, subdomain_id="everywhere"))
else:
    meshData['fluid']['initVol'] = float(args.volCons)
meshData['fluid']['initNumCells'] = mesh.num_cells()
flow_direction = Constant((1.0,0.0))
justRemeshed = True
krylov_iters = 0
solution_time = 0.0

meshFile = File(args.out_folder+"/mesh.pvd")
uFile = File(args.out_folder+"/velocity.pvd")
pFile = File(args.out_folder+"/pressure.pvd")
adj_uFile = File(args.out_folder+"/adj_velocity.pvd")
adj_pFile = File(args.out_folder+"/adj_pressure.pvd")
tFile = File(args.out_folder+"/temperature.pvd")
adj_tFile = File(args.out_folder+"/adj_temperature.pvd")
vFile = File(args.out_folder+"/shape_gradient.pvd")

#info("Courant number: Co = %g ~ %g" % (u0*args.dt/mesh.hmax(), u0*args.dt/mesh.hmin()))

##################################
####         MAIN PART        ####
##################################

for iterNo in range(systemPara['maxIter']):

    info('############################################')
    info('# A new iteration, iteration {:g} has begun! #'.format(iterNo+1))
    info('############################################')

    if justRemeshed:
        
        justRemeshed = False
        info('**********************************************')
        info('Forming problems and solvers for the new mesh!')
        info('**********************************************')

        meshData['fluid']['subdomain'] = mU.markSubDomains(mesh)
        subDomain_markers = meshData['fluid']['subdomain']
        meshData['fluid']['boundary'] = mU.markBoundaries(mesh)
        boundary_markers = meshData['fluid']['boundary']
        meshData['fluid']['n'] = FacetNormal(mesh)
        meshData['fluid']['h'] = CellDiameter(mesh)
        meshData['fluid']['dx'] = Measure("dx", domain=mesh, subdomain_id="everywhere")
        meshData['fluid']['dX'] = Measure("dx", domain=mesh, subdomain_data=subDomain_markers)
        meshData['fluid']['ds'] = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
        try: 
            meshData['fluid']['bndVIDs'] = mU.getSeedVerticesFromPts(meshData, 'fluid')
        except:
            meshData['fluid']['bndVIDs'] = []
            info('!!!!! Failed to extract boundary vertices, may not be able to auto-remesh !!!!!')

        pbc = None#gU.yPeriodic(mapFrom=0.0, mapTo=1.6)
        Vec2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Vec1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
        Sca1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        Real0 = FiniteElement("R", mesh.ufl_cell(), 0)
        meshData['fluid']['spaceNS'] = FunctionSpace(mesh, MixedElement([Vec2, Sca1]), constrained_domain=pbc)
        meshData['fluid']['spaceThermal'] = FunctionSpace(mesh, Sca1, constrained_domain=pbc)
        meshData['fluid']['spaceSG'] = FunctionSpace(mesh, MixedElement([Vec1, Real0]))
        meshData['fluid']['spaceLE'] = FunctionSpace(mesh, Vec1) # LE=LinearElasticity, used for mesh moving
        SG2LEAssigner = FunctionAssigner(meshData['fluid']['spaceLE'], meshData['fluid']['spaceSG'].sub(0)) # this assigner extracts mesh velocity and drops Lag. multi.
        
        funcVar['fluid']['up'] = Function(meshData['fluid']['spaceNS']) 
        funcVar['fluid']['up_prime'] = Function(meshData['fluid']['spaceNS']) # _prime marks adjoint variables 
        funcVar['fluid']['T'] = Function(meshData['fluid']['spaceThermal']) 
        funcVar['fluid']['T_prime'] = Function(meshData['fluid']['spaceThermal']) 
        funcVar['fluid']['v'] = Function(meshData['fluid']['spaceSG'])  # shape grad, or say mesh velocity, as well as Lag. multi.
        funcVar['fluid']['modified_v'] = Function(meshData['fluid']['spaceLE'])  # scaled or otherwise processed shape grad
        funcVar['fluid']['w'] = Function(meshData['fluid']['spaceLE'])  # final mesh move direction; used in ALE

        BCs['fluid']['NS'] = mU.applyNSBCs(meshData, boundary_markers)
        BCs['fluid']['adjNS'] = mU.applyAdjNSBCs(meshData, boundary_markers)
        BCs['fluid']['thermal'] = mU.applyThermalBCs(meshData, boundary_markers)
        BCs['fluid']['adjThermal'] = mU.applyAdjThermalBCs(meshData, boundary_markers) 
        BCs['fluid']['SG'] = mU.applyShapeGradientBCs(meshData, boundary_markers)
        BCs['fluid']['LE'] = mU.applyLinearElasticityBCs(meshData, boundary_markers, funcVar, physicalPara)

        problemNS = sS.formProblemNS(meshData, BCs, physicalPara, funcVar, systemPara)
        problemAdjNS = sS.formProblemAdjNS(meshData, BCs, physicalPara, funcVar, systemPara)
        problemThermal = sS.formProblemThermal(meshData, BCs, physicalPara, funcVar, systemPara) 
        problemAdjThermal = sS.formProblemAdjThermal(meshData, BCs, physicalPara, funcVar, systemPara)

        solverNS = sS.formSolverNS(problemNS, systemPara)
        solverAdjNS = sS.formSolverNS(problemAdjNS, systemPara) # Adj NS former is the same as the state NS former
        solverThermal = sS.formSolverThermal(problemThermal, systemPara)
        solverAdjThermal = sS.formSolverThermal(problemAdjThermal, systemPara)

        # now form the problem solving for the shape gradient
        problemSG = sS.formProblemShapeGradient(meshData, BCs, physicalPara, funcVar, systemPara)
        solverSG = sS.formSolverShapeGradient(problemSG, systemPara)   
        # now form the problem using linear elasiticity to move the mesh
        problemLE = sS.formProblemLinearElasticity(meshData, BCs, physicalPara, funcVar, systemPara)
        solverLE = sS.formSolverShapeGradient(problemLE, systemPara) # SG and LE are both simple linear problems, can use the same former 

        info('****************************************')
        info('Problems and solvers sucessfully formed!')
        info('****************************************')
    ########### End of mesh setup and problem definition ###############

    ########### Begining solving systems ###############################
    info('------------------------------')
    info("Begining to solve systems...")
    info('------------------------------')

    if systemPara['ns'] == "rmturs":
        with Timer("SolveSystems") as t_solve:
            newton_iters, converged = solverNS.solve(problemNS, funcVar['fluid']['up'].vector())
    elif systemPara['ns'] == "variational":
        with Timer("SolveSystems") as t_solve:
            solverNS.solve()
            solverThermal.solve()
            solverAdjThermal.solve()
            solverAdjNS.solve()
            solverSG.solve()

    #krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    (objHeat, objDissp) = sS.computeObj(meshData, physicalPara, funcVar)
    outputData['objHeat'].append(objHeat)
    outputData['objDissp'].append(objDissp)

    if (iterNo+1) % systemPara['ts_per_out']==0:
        u_out, p_out = funcVar['fluid']['up'].split()
        adj_u_out, adj_p_out = funcVar['fluid']['up_prime'].split()
        uFile << (u_out, iterNo)
        pFile << (p_out, iterNo)
        adj_uFile << (adj_u_out, iterNo)
        adj_pFile << (adj_p_out, iterNo)
        tFile << (funcVar['fluid']['T'], iterNo)
        adj_tFile << (funcVar['fluid']['T_prime'], iterNo)
        vFile << (funcVar['fluid']['v'], iterNo)
        meshFile << (mesh, iterNo)

    # now move the mesh and remesh (if needed)
    SG2LEAssigner.assign(funcVar['fluid']['modified_v'], funcVar['fluid']['v'].sub(0))
    funcVar['fluid']['modified_v'].vector()[:] = physicalPara['stepLen']*funcVar['fluid']['modified_v'].vector()[:]
    solverLE.solve()
    ALE.move(mesh, funcVar['fluid']['w'])

    if (iterNo+1) % systemPara['ts_per_rm']==0:
        meshData['fluid']['bndExPts'] = mU.getSeedPtsFromVertices(meshData, 'fluid')
        meshData['fluid']['mesh'] = mU.createMeshViaTriangle(meshData, 'fluid') 
        mesh = meshData['fluid']['mesh']
        assert mesh is not None
        justRemeshed = True

# Report timings
list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Get iteration counts
result = {
    "ndof": meshData['fluid']['spaceNS'].dim(), "time": solution_time, "steps": iterNo+1,
    "lin_its": krylov_iters, "lin_its_avg": float(krylov_iters)/(iterNo+1)}
tab = "{:^15} | {:^15} | {:^15} | {:^19} | {:^15}\n".format(
    "No. of DOF", "Steps", "Krylov its", "Krylov its (p.t.s.)", "Time (s)")
tab += "{ndof:>9}       | {steps:^15} | {lin_its:^15} | " \
       "{lin_its_avg:^19.1f} | {time:^15.2f}\n".format(**result)
print("\nSummary of iteration counts:")
print(tab)

print("-----------------------------------------------------------------")
print("objHeat: ", outputData['objHeat'])
print("objDissp: ", outputData['objDissp']) 
#with open("table_pcdr_{}.txt".format(args.pcd_variant), "w") as f:
#    f.write(tab)

#rgs.out_folder+"/pressure.pvd") Plot solution
#u, p = w.split()
#size = MPI.size(mesh.mpi_comm())
#rank = MPI.rank(mesh.mpi_comm())
#pyplot.figure()
#pyplot.subplot(2, 1, 1)
#plot(u, title="velocity")
#pyplot.subplot(2, 1, 2)
#plot(p, title="pressure")
#pyplot.savefig("figure_v_p_size{}_rank{}.pdf".format(size, rank))
#pyplot.figure()
#plot(p, title="pressure", mode="warp")
#pyplot.savefig("figure_warp_size{}_rank{}.pdf".format(size, rank))
#pyplot.show()
