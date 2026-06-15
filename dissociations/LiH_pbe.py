#ANALIZAMOS PBE in HF
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, dft, cc

distancias=np.linspace(0.4,3.5,30)

energias_fci = []
energias_dft = []


energia_dif_dft = []

for R in distancias:
    mol= gto.M(
        atom=f'''
        Li   0.0000  0.0000   0.0000
        H   0.0000  0.0000   {R} 
        ''',
        basis='def2-TZVP', 
        unit='Angstrom'
    )

    ml = dft.RKS(mol)
    ml.xc = 'pbe'
    e_dft = ml.kernel()

    cisolver = fci.FCI(mol, ml.mo_coeff)
    e_fci, ci_vector = cisolver.kernel()

   
    energias_fci.append(e_fci)
    energias_dft.append(e_dft)
    

    
    energia_dif_dft.append(e_fci-e_dft)
    
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Primer gráfico

ax[0].plot(distancias, energias_dft, "o-", label="PBE")

ax[0].plot(distancias, energias_fci, "-", color="black", linewidth=2.5, label="FCI in PBE")

ax[0].set_xlabel("Distancia Li-H (Å)")
ax[0].set_ylabel("Energía (Hartree)")
ax[0].legend()
ax[0].grid()

# Segundo gráfico
# Aquí pones lo que quieras

ax[1].plot(distancias, energia_dif_dft, "o-", label="PBE")



ax[1].set_xlabel("Distancia Li-H (Å)")
ax[1].set_ylabel(r"$\Delta E_{\mathrm{FCI}}$ (Hartree)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()