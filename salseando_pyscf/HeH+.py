from pyscf import gto, scf, ao2mo
import numpy as np

from openfermion import InteractionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner

mol = gto.M(
    atom="He 0 0 0; H 0 0 0.772",
    basis="sto-3g",
    unit="Angstrom",
    charge=1,
    spin=0
)

mf = scf.RHF(mol)
E_hf = mf.kernel()

hcore_ao = mf.get_hcore()
C = mf.mo_coeff
h1_mo = C.T @ hcore_ao @ C

eri_mo = ao2mo.kernel(mol, C)
norb = C.shape[1]
h2_mo = ao2mo.restore(1, eri_mo, norb)

nuclear_repulsion = mol.energy_nuc()

fermion_hamiltonian = get_fermion_operator(
    InteractionOperator(
        nuclear_repulsion,
        h1_mo,
        h2_mo
    )
)

qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

print(qubit_hamiltonian)