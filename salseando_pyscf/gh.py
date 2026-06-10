from pyscf import gto, scf, mcscf

from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_nature_pyscf import QiskitSolver


mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")

h_f = scf.RHF(mol).run()

norb, nelec = 2, 2

cas = mcscf.CASCI(h_f, norb, nelec)

mapper = JordanWignerMapper()
solver = NumPyMinimumEigensolver()

algorithm = GroundStateEigensolver(mapper, solver)

cas.fcisolver = QiskitSolver(algorithm)

cas.run()