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
    def __init__(self, a, L, bcs):
        self.assembler = SystemAssembler(a, L, bcs)
        self._bcs = bcs
    def rhs_vector(self, b, x=None):
        if x is not None:
            self.assembler.assemble(b, x)
        else:
            self.assembler.assemble(b)
    def system_matrix(self, A):
        self.assembler.assemble(A)

class rmtursNonlinearProblem(NonlinearProblem):
    def __init__(self, rmturs_assembler):
        #assert isinstance(rmturs_assembler, rmtursAssembler)
        super(rmtursNonlinearProblem, self).__init__()
        self.rmturs_assembler = rmturs_assembler
    def F(self, b, x):
        self.rmturs_assembler.rhs_vector(b, x)
    def J(self, A, x):
        self.rmturs_assembler.system_matrix(A)

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


