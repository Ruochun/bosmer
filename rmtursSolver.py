from dolfin import SystemAssembler, assemble, NewtonSolver, PETScFactory, NonlinearProblem, compile_cpp_code

_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/nls/NewtonSolver.h>
PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<dolfin::NewtonSolver, std::shared_ptr<dolfin::NewtonSolver>>
  (m, "NewtonSolverExt", pybind11::module_local())
  .def("krylov_iterations", &dolfin::NewtonSolver::krylov_iterations);
}
"""
NewtonSolver.krylov_iterations = compile_cpp_code(_cpp).NewtonSolverExt.krylov_iterations

class rmtursAssembler(object):
    def __init__(self, a, L, bcs, a_pc=None):
        self.assembler = SystemAssembler(a, L, bcs)
        if a_pc is not None:
            self.assembler_pc = SystemAssembler(a_pc, L, bcs)
        else:
            self.assembler_pc = None
        self._bcs = bcs
    def rhs_vector(self, b, x=None):
        if x is not None:
            self.assembler.assemble(b, x)
        else:
            self.assembler.assemble(b)
    def system_matrix(self, A):
        self.assembler.assemble(A)
    def pc_matrix(self, P):
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)
    
class rmtursNonlinearProblem(NonlinearProblem):
    def __init__(self, rmturs_assembler):
        #assert isinstance(rmturs_assembler, rmtursAssembler)
        super(rmtursNonlinearProblem, self).__init__()
        self.rmturs_assembler = rmturs_assembler
    def F(self, b, x):
        self.rmturs_assembler.rhs_vector(b, x)
    def J(self, A, x):
        self.rmturs_assembler.system_matrix(A)
    def J_pc(self, P, x):
        self.rmturs_assembler.pc_matrix(P)

class rmtursNewtonSolver(NewtonSolver):
    def __init__(self, solver):
        comm = solver.ksp().comm.tompi4py()
        factory = PETScFactory.instance()
        super(rmtursNewtonSolver, self).__init__(comm, solver, factory)
        self._solver = solver
    def solve(self, problem, x):
        self._problem = problem
        r = super(rmtursNewtonSolver, self).solve(problem, x)
        del self._problem
        return r
    def linear_solver(self):
        return self._solver
    def solver_setup(self, A, P, nonlinear_problem, iteration):
        if iteration>0 or getattr(self, "_initialized", False):
            return
        self._initialized = True
        linear_solver = self._solver
        nonlinear_problem = self._problem
        P = A if P.empty() else P
        linear_solver.set_operators(A, P)



