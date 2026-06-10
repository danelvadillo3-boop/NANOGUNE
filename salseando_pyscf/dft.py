from pyscf import dft, gto

mol=gto.M(atom= 'O 0 0 0; H 0 1 0; H 1 0 0', basis='cc-pvdz', charge=0, spin=0)


rks_h2o = dft.RKS(mol) # likewise for UKS and ROKS
rks_h2o.xc = 'b3lyp' #es una aproximación del termino XC
rks_h2o.kernel()
# 1. Definimos la molécula de Metano (geometría típica en Angstroms)
mol_ch4 = gto.M(
    atom='''
    C    0.0000    0.0000    0.0000
    H    0.6291    0.6291    0.6291
    H   -0.6291   -0.6291    0.6291
    H    0.6291   -0.6291   -0.6291
    H   -0.6291    0.6291   -0.6291
    ''',
    basis='6-31g',  # Una base estándar muy usada
)

# 2. Preparamos el cálculo DFT con el funcional B3LYP
mf = dft.RKS(mol_ch4)
mf.xc = 'b3lyp'

# 3. Ejecutamos el cálculo de la energía
energia = mf.kernel()
print(f"\nEnergía final (B3LYP) para metano: {energia} Hartrees")

# 4. Calculamos el momento dipolar
dipolo = mf.dip_moment()