from pyscf import gto, scf, mcscf

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")

h_f = scf.RHF(mol).run()

norb, nelec = 2, 2

cas = mcscf.CASCI(h_f, norb, nelec)

cas.run()