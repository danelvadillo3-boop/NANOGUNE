from pyscf import dft, gto
mol_tio2 = gto.M(
    atom='''
    Ti   0.0000  0.0000   0.0000
    O    1.620   0.0000   0.0000
    O   -1.620   0.0000   0.0000
    ''',
    basis='def2-TZVP', 
    unit='Angstrom'
)

# 2. Preparamos el cálculo DFT con el funcional
mf = dft.RKS(mol_tio2)
mf.xc = 'pbe0'

# 3. Ejecutamos el cálculo de la energía
energia = mf.kernel()