from pyscf import gto, scf, solvent

mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    O  1.4300  0.0000  0.0000
    H -0.5400  0.9350  0.0000
    H -0.5400 -0.4675  0.8095
    H -0.5400 -0.4675 -0.8095
    H  1.9000  0.0000  0.8000
    """,
    basis="cc-pvdz",
    charge=0,
    spin=0,
    unit="Angstrom"
)

mf = scf.RHF(mol)
mf_water = solvent.ddPCM(mf)
#mf_water.with_solvent.eps = 1000
energia_con_agua = mf_water.kernel()
mf_prima = scf.HF(mol)
energia_sin_agua = mf_prima.kernel()

print(energia_con_agua/energia_sin_agua)