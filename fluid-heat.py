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
parser.add_argument("-l", type=int, dest="level", default=0,
                    help="level of mesh refinement")
parser.add_argument("--nu", type=float, dest="viscosity", default=0.02,
                    help="kinematic viscosity")
parser.add_argument("--Pe", type=float, dest="Pe", default=10.,
                    help="Peclet number for thermal system")
parser.add_argument("--ls", type=str, dest="ls", default="iterative",
                    choices=["direct", "iterative"], help="linear solvers")
parser.add_argument("--ts_per_out", type=int, dest="ts_per_out", default=1,
                    help="number of ts per output file")
parser.add_argument("--mesh_file", type=str, dest="mesh_file", default="__SAMPLE",
                    help="path and file name of the mesh, or do not specify to use the defualt sample mesh")
parser.add_argument("--out_folder", type=str, dest="out_folder", default="./result",
                    help="output folder name")
parser.add_argument("--maxIter", type=int, dest="maxIter", default=10,
                    help="total number of iteration steps")
args = parser.parse_args(sys.argv[1:])

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["std_out_all_processes"] = False
rank = commmpi.Get_rank()
root = 0
# Load mesh from file and refine uniformly
meshData = {} # mesh and related info
BCs = {} # BC sets
funcVar = {} # state and adjoint variables
physicalPara = {} # physical parameters of this problem

maxIter = args.maxIter
physicalPara['fluid']['nu'] = args.viscosity
physicalPara['fluid']['Pe'] = args.Pe

if mesh_file == "__SAMPLE":
    meshData['fluid']['mesh'] = mU.sampleMesh(100)
    mesh = meshData['fluid']['mesh']
    boundary_points = [1.5, .5]
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
justRemeshed = False
##################################
####         MAIN PART        ####
##################################

for iterNo in range(maxIter):

    info('*****************************')
    info('* Beginning a new iteration *')
    info('*****************************')

    if (iterNo==0) or (justRemeshed):
        
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
        Sca1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W_NS = FunctionSpace(mesh, MixedElement([Vec2, Sca1]), constrained_domain=pbc)
        W_thermal = FunctionSpace(mesh, Sca1, constrained_domain=pbc)

        BCs['fluid']['NS'] = mU.applyNSBCs(W_NS, boundary_markers)
        BCs['fluid']['adjNS'] = mU.applyAdjNSBCs(W_NS, boundary_markers)
        BCs['fluid']['thermal'] = mU.applyThermalBCs(W_thermal, boundary_markers)
        BCs['fluid']['adjThermal'] = mU.applyAdjThermalBCs(W_thermal, boundary_markers) 

        funcVar['fluid']['up'] = Function(W_NS)
        funcVar['fluid']['up_prime'] = Function(W_NS) # _prime marks adjoint variables
        funcVar['fluid']['T'] = Function(W_thermal)
        funcVar['fluid']['T_prime'] = Function(W_thermal)

        problemNS = sS.formProblemNS(meshData, W_NS, BCs, physicalPara, funcVar)
        problemAdjNS = sS.formProblemAdjNS(meshData, W_NS, BCs, physicalPara, funcVar)
        problemThermal = sS.formProblemThermal(W_thermal, BCs, physicalPara, funcVar) 
        problemAdjThermal = sS.formProblemAdjThermal(W_thermal, BCs, physicalPara, funcVar)

        info("Courant number: Co = %g ~ %g" % (u0*args.dt/mesh.hmax(), u0*args.dt/mesh.hmin()))

        # Set up linear solver (GMRES with right preconditioning using Schur fact)
        PETScOptions.clear()
        linear_solver = PETScKrylovSolver()
        linear_solver.parameters["relative_tolerance"] = 1e-5
        linear_solver.parameters["absolute_tolerance"] = 1e-12
        linear_solver.parameters['error_on_nonconvergence'] = False
        PETScOptions.set("ksp_monitor")

        # Set up subsolvers
        if args.ls == "iterative":
            PETScOptions.set("ksp_type", "fgmres")
            PETScOptions.set("ksp_gmres_restart", 10)
            PETScOptions.set("ksp_max_it", 100)
            PETScOptions.set("preconditioner", "default")
            #PETScOptions.set("nonzero_initial_guess", True)

        # Apply options
        linear_solver.set_from_options()

        # Set up nonlinear solver
        solver = rmtursNewtonSolver(linear_solver)
        solver.parameters["relative_tolerance"] = 1e-3
        solver.parameters["error_on_nonconvergence"] = False
        solver.parameters["maximum_iterations"] = 7
        if rank == 0:
            set_log_level(20) #INFO level, no warnings
        else:
            set_log_level(50)
    # files
    #if rank == 0:
    ufile = File(args.out_folder+"/velocity.pvd")
    pfile = File(args.out_folder+"/pressure.pvd")
    # Solve problem
    t = 0.0
    time_iters = 0
    krylov_iters = 0
    solution_time = 0.0

    info("t = {:g}, step = {:g}, dt = {:g}".format(t, time_iters, args.dt))
    with Timer("Solve") as t_solve:
        newton_iters, converged = solver.solve(problem, w.vector())
    krylov_iters += solver.krylov_iterations()
    solution_time += t_solve.stop()

    if (iterNo % args.ts_per_out==0)or(iterNo == 0):
        ufile << u_out
        pfile << p_out


# Report timings
list_timings(TimingClear.clear, [TimingType.wall, TimingType.user])

# Get iteration counts
result = {
    "ndof": W.dim(), "time": solution_time, "steps": time_iters,
    "lin_its": krylov_iters, "lin_its_avg": float(krylov_iters)/time_iters}
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
