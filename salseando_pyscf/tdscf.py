from pyscf import gto, scf, dft, tddft, tdscf
mol=gto.M(atom= 'O 0 0 0; H 0 1 0; H 1 0 0', basis='cc-pvdz', charge=0, spin=0)
mf = scf.HF(mol)
#energia_final = mf.kernel()
rhf_h2o = scf.RHF(mol)
rks_h2o = dft.RKS(mol)
tdhf_h2o =tdscf.TDHF(rhf_h2o)
tdhf_h2o.nstates = 6
tdhf_h2o.kernel()
tddft_h2o = tdscf.TDA(rks_h2o) # TDDFT with Tamm-Dancoff approximation
tddft_h2o.nstates = 4
tddft_h2o.kernel()