import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, cc, mcscf

distancias = np.linspace(0.3, 1.3, 50)

E_fci_in_hf_AS22 = []
E_fci_in_hf_AS44 = []
E_ccsd_in_hf_AS22 = []
E_ccsd_in_hf_AS44 = []
ground_energy= 0.0
for R in distancias:
    mol = gto.M(
        atom=f"H 0 0 {-R}; C 0 0 0; H 0 0 {R}",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0
    )

    mf = scf.RHF(mol).run()

    nmo = mf.mo_coeff.shape[1]

    # ---------- AS(2,2) ----------
    cas22 = mcscf.CASCI(mf, 2, 2)
    e_fci_as22 = cas22.kernel()[0]

    if e_fci_as22 < ground_energy :
        ground_energy= e_fci_as22
        r_min=R

    E_fci_in_hf_AS22.append(e_fci_as22)

    ncas = 2
    nelecas = 2
    ncore = (mol.nelectron - nelecas) // 2

    active_orbs = list(range(ncore, ncore + ncas))
    frozen = [i for i in range(nmo) if i not in active_orbs]

    mycc22 = cc.CCSD(mf, frozen=frozen)
    mycc22.kernel()
    E_ccsd_in_hf_AS22.append(mycc22.e_tot)
    
    # ---------- AS(4,4) ----------
    cas44 = mcscf.CASCI(mf, 4, 4)
    e_fci_as44 = cas44.kernel()[0]
    E_fci_in_hf_AS44.append(e_fci_as44)

    ncas = 4
    nelecas = 4
    ncore = (mol.nelectron - nelecas) // 2

    active_orbs = list(range(ncore, ncore + ncas))
    frozen = [i for i in range(nmo) if i not in active_orbs]

    mycc44 = cc.CCSD(mf, frozen=frozen)
    mycc44.kernel()
    E_ccsd_in_hf_AS44.append(mycc44.e_tot)

print("Ground state energy:",ground_energy, "R_min:", r_min)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# AS(2,2)
ax[0].plot(distancias, E_fci_in_hf_AS22, "-", color="black",
           linewidth=2.5, label="FCI-in-HF")
ax[0].plot(distancias, E_ccsd_in_hf_AS22, "o-", color="red",
           markersize=4, label="CCSD-in-HF")

ax[0].set_title("AS(2,2)")
ax[0].set_xlabel("Distancia C-H (Å)")
ax[0].set_ylabel("Energía (Hartree)")
ax[0].legend()
ax[0].grid()

# AS(4,4)
ax[1].plot(distancias, E_fci_in_hf_AS44, "-", color="black",
           linewidth=2.5, label="FCI-in-HF")
ax[1].plot(distancias, E_ccsd_in_hf_AS44, "o-", color="red",
           markersize=4, label="CCSD-in-HF")

ax[1].set_title("AS(4,4)")
ax[1].set_xlabel("Distancia C-H (Å)")
ax[1].set_ylabel("Energía (Hartree)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig("ch2_active_space.png", dpi=300)
plt.show()