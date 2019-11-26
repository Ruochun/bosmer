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
parser.add_argument("-l", type=int, dest="level", default=0,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.2,
                    help="kinematic viscosity")
parser.add_argument("--Pe", type=float, dest="Pe", default=10.,
                    help="Peclet number for thermal system")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solver, choose from direct or iterative")
parser.add_argument("--ts_per_out", type=int, dest="ts_per_out", default=1,
                    help="number of ts per output file")
parser.add_argument("--mesh_file", type=str, dest="mesh_file", default="__SAMPLE",
                    help="path and file name of the mesh, or do not specify to use the defualt sample mesh")
parser.add_argument("--out_folder", type=str, dest="out_folder", default="./result",
                    help="output folder name")
parser.add_argument("--max_iter", type=int, dest="max_iter", default=10,
                    help="total number of iteration steps")
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

# we have a fluid system
meshData['fluid'] = {}
BCs['fluid'] = {}
funcVar['fluid'] = {}
physicalPara['fluid'] = {}
# we don't currently have a solid system


# load in system settings
maxIter = args.max_iter
physicalPara['fluid']['nu'] = args.viscosity
physicalPara['fluid']['Pe'] = args.Pe
systemPara['ns'] = args.ns
systemPara['ls'] = args.ls

if args.mesh_file == "__SAMPLE":
    meshData['fluid']['mesh'] = mU.sampleMesh()
    mesh = meshData['fluid']['mesh']
    boundary_points = [1., .5]
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
    boundary_points = []

    for i in range(args.level):
        mesh = refine(mesh)

flow_direction = Constant((1.0,0.0))
justRemeshed = True
##################################
####         MAIN PART        ####
##################################

for iterNo in range(maxIter):

    info('*****************************')
    info('* Begining a new iteration *')
    info('*****************************')

    if justRemeshed:
        
        justRemeshed = False
        info('####################################################################')
        info('Setting up function spaces and problem definitions for the new mesh!')
        info('####################################################################')

        meshData['fluid']['subdomain'] = mU.markSubDomains(mesh)
        subDomain_markers = meshData['fluid']['subdomain']
        meshData['fluid']['boundary'] = mU.markBoundaries(mesh)
        boundary_markers = meshData['fluid']['boundary']
        meshData['fluid']['n'] = FacetNormal(mesh)
        meshData['fluid']['h'] = CellDiameter(mesh)
        meshData['fluid']['dx'] = Measure("dx", domain=mesh, subdomain_id="everywhere")
        meshData['fluid']['dX'] = Measure("dx", domain=mesh, subdomain_data=subDomain_markers)
        meshData['fluid']['ds'] = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

        # Build function spaces
        pbc = gU.yPeriodic(mapFrom=0.0, mapTo=1.0)
        Vec2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Vec1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
        Sca1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        Real0 = FiniteElement("R", mesh.ufl_cell(), 0)
        meshData['fluid']['spaceNS'] = FunctionSpace(mesh, MixedElement([Vec2, Sca1]), constrained_domain=pbc)
        meshData['fluid']['spaceThermal'] = FunctionSpace(mesh, Sca1, constrained_domain=pbc)
        meshData['fluid']['spaceSG'] = FunctionSpace(mesh, MixedElement([Vec1, Real0]))
        #meshData['fluid']['spaceLE'] = FunctionSpace(mesh, Vec1)

        BCs['fluid']['NS'] = mU.applyNSBCs(meshData, boundary_markers)
        BCs['fluid']['adjNS'] = mU.applyAdjNSBCs(meshData, boundary_markers)
        BCs['fluid']['thermal'] = mU.applyThermalBCs(meshData, boundary_markers)
        BCs['fluid']['adjThermal'] = mU.applyAdjThermalBCs(meshData, boundary_markers) 

        funcVar['fluid']['up'] = Function(meshData['fluid']['spaceNS'])
        funcVar['fluid']['up_prime'] = Function(meshData['fluid']['spaceNS']) # _prime marks adjoint variables
        funcVar['fluid']['T'] = Function(meshData['fluid']['spaceThermal'])
        funcVar['fluid']['T_prime'] = Function(meshData['fluid']['spaceThermal'])
        funcVar['fluid']['v'] = Function(meshData['fluid']['spaceSG'])
        #funcVar['fluid']['w'] = Function(meshData['fluid']['spaceLE'])

        problemNS = sS.formProblemNS(meshData, BCs, physicalPara, funcVar, systemPara)
        problemAdjNS = sS.formProblemAdjNS(meshData, BCs, physicalPara, funcVar, systemPara)
        problemThermal = sS.formProblemThermal(meshData, BCs, physicalPara, funcVar, systemPara) 
        problemAdjThermal = sS.formProblemAdjThermal(meshData, BCs, physicalPara, funcVar, systemPara)

        solverNS = sS.formSolverNS(problemNS, systemPara)
        solverAdjNS = sS.formSolverNS(problemAdjNS, systemPara) # Adj NS former is the same as the state NS former
        solverThermal = sS.formSolverThermal(problemThermal, systemPara)
        solverAdjThermal = sS.formSolverThermal(problemAdjThermal, systemPara)
        #info("Courant number: Co = %g ~ %g" % (u0*args.dt/mesh.hmax(), u0*args.dt/mesh.hmin()))
    
    ########### End of mesh setup and problem definition ###############
    ########### Begining solving systems ###############################
    meshFile = File(args.out_folder+"/mesh.pvd") 
    uFile = File(args.out_folder+"/velocity.pvd")
    pFile = File(args.out_folder+"/pressure.pvd")
    adj_uFile = File(args.out_folder+"/adj_velocity.pvd")
    adj_pFile = File(args.out_folder+"/adj_pressure.pvd")
    tFile = File(args.out_folder+"/temperature.pvd")
    adj_tFile = File(args.out_folder+"/adj_temperature.pvd")


    # Solve problem
    krylov_iters = 0
    solution_time = 0.0

    info("step = {:g}".format(iterNo+1))
    if systemPara['ns'] == "rmturs":
        with Timer("SolveNS") as t_solve:
            newton_iters, converged = solverNS.solve(problemNS, funcVar['fluid']['up'].vector())
    else:
        with Timer("SolveNS") as t_solve:
            solverNS.solve()
            solverAdjNS.solve()
            solverThermal.solve()
            solverAdjThermal.solve()

    #krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()
    
    #(w, lam) = sS.searchDirection(meshData, funcVar)
    if (iterNo % args.ts_per_out==0):
        u_out, p_out = funcVar['fluid']['up'].split()
        adj_u_out, adj_p_out = funcVar['fluid']['up_prime'].split()
        uFile << u_out
        pFile << p_out
        adj_uFile << adj_u_out
        adj_pFile << adj_p_out
        tFile << funcVar['fluid']['T']
        adj_tFile << funcVar['fluid']['T_prime']
        meshFile << mesh

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
#with open("table_pcdr_{}.txt".format(args.pcd_variant), "w") as f:
#    f.write(tab)

#rgs.out_folder+"/pressure.pvd") Plot solution
#u, p = w.split()
#size = MPI.size(mesh.mpi_comm())
#rank = MPI.rank(mesh.mpi_comm())
"""
pyplot.figure()
pyplot.subplot(2, 1, 1)
plot(u, title="velocity")
pyplot.subplot(2, 1, 2)
plot(p, title="pressure")
pyplot.savefig("figure_v_p_size{}_rank{}.pdf".format(size, rank))
pyplot.figure()
plot(p, title="pressure", mode="warp")
pyplot.savefig("figure_warp_size{}_rank{}.pdf".format(size, rank))
pyplot.show()
"""
