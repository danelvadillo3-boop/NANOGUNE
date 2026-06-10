import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, dft, cc

distancias=np.linspace(0.4,3.5,30)
energias_hf = []
energias_fci = []
energias_dft = []
energias_ccsd = []
energia_dif_hf = []
energia_dif_dft = []
energia_dif_ccsd = []
for R in distancias:
    mol= gto.M(
        atom=f'''
        F   0.0000  0.0000   0.0000
        F   0.0000  0.0000   {R} 
        ''',
        basis='def2-TZVP', 
        unit='Angstrom'
    )

    mf=scf.RHF(mol)
    e_hf=mf.kernel()

    mk=cc.CCSD(mf)
    e_ccsd=e_hf + mk.kernel()[0] #Cuidado, mk es la energía de correlación

    ml = dft.RKS(mol)
    ml.xc = 'b3lyp'
    e_dft = ml.kernel()

    cisolver = fci.FCI(mol, mf.mo_coeff)
    e_fci, ci_vector = cisolver.kernel()

    energias_hf.append(e_hf)
    energias_fci.append(e_fci)
    energias_dft.append(e_dft)
    energias_ccsd.append(e_ccsd)

    energia_dif_hf.append(e_fci - e_hf)
    energia_dif_dft.append(e_fci-e_dft)
    energia_dif_ccsd.append(e_fci-e_ccsd)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Primer gráfico
ax[0].plot(distancias, energias_hf, "o-", label="HF")
ax[0].plot(distancias, energias_dft, "o-", label="DFT")
ax[0].plot(distancias, energias_ccsd, "o", color="red", markersize=5, label="CCSD")
ax[0].plot(distancias, energias_fci, "-", color="black", linewidth=2.5, label="FCI")

ax[0].set_xlabel("Distancia Li-H (Å)")
ax[0].set_ylabel("Energía (Hartree)")
ax[0].legend()
ax[0].grid()

# Segundo gráfico
# Aquí pones lo que quieras
ax[1].plot(distancias, energia_dif_hf, "o-", label="HF")
ax[1].plot(distancias, energia_dif_dft, "o-", label="DFT")
ax[1].plot(distancias, energia_dif_ccsd, "o", color="red", markersize=5, label="CCSD")


ax[1].set_xlabel("Distancia F-F (Å)")
ax[1].set_ylabel(r"$\Delta E_{\mathrm{FCI}}$ (Hartree)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()