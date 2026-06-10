from pyscf import gto, scf, dft, tddft, tdscf
h = 6.62607015e-34        # J s
c = 2.99792458e8          # m/s
hartree_to_joule = 4.3597447222071e-18  
mol=gto.M(atom= 'N 0 0 0; N 0 0 1.1', basis='cc-pvdz', unit='Angstrom', charge=0, spin=0)
mf = dft.RKS(mol)
mf.xc = 'b3lyp'

# 3. Ejecutamos el cálculo de la energía
energia = mf.kernel()
tddft_n2 = tdscf.TDA(mf) # TDDFT with Tamm-Dancoff approximation
tddft_n2.nstates = 4
tddft_n2.kernel()


E_hartree = tddft_n2.e[0]
E_joule = E_hartree * hartree_to_joule
lambda_m = h * c / E_joule
lambda_nm = lambda_m * 1e9
print(lambda_nm)