from pyscf import gto, scf, solvent

mol = gto.M(
    atom='''
    N  0.000  0.000  0.000
    H  0.000  0.000  1.040
    H  0.981  0.000 -0.347
    H -0.491  0.850 -0.347
    H -0.491 -0.850 -0.347
    ''',
    basis='cc-pvdz',
    charge=1,
    spin=0
)

# Gas
mf_gas = scf.RHF(mol)
Egas = mf_gas.kernel()

# PCM
mf_pcm = solvent.ddPCM(scf.RHF(mol))
mf_pcm.with_solvent.eps = 78.4
Epcm = mf_pcm.kernel()

print("Egas =", Egas)
print("Epcm =", Epcm)
print("Diferencia =", Epcm - Egas)