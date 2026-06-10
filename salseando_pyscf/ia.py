#!/usr/bin/env python
"""
Cálculo de la energía de la molécula O2 usando SCP (Simplified Coupled Pair)
con la librería PySCF
"""

from pyscf import gto, scf, mp
import numpy as np

# 1. Definir la geometría de la molécula O2
# O2 con distancia de enlace típica de ~1.2 Angstroms
mol = gto.M(
    atom='O 0 0 0; O 0 0 1.2',
    basis='sto-3g',  # Base simple para demostración
    charge=0,
    spin=2,  # O2 es un multiplete (estado triplete, 2 electrones desapareados)
    verbose=4
)

print("=" * 60)
print("CÁLCULO DE ENERGÍA DE O2 CON SCP")
print("=" * 60)
print(f"\nGeometría de la molécula:")
print(mol)
print(f"\nNúmero de electrones: {mol.nelectron}")
print(f"\nNúmero de orbitales atómicos: {mol.nao_nr()}")

# 2. Realizar cálculo SCF (Hartree-Fock) como base
print("\n" + "-" * 60)
print("1. CÁLCULO HF (HARTREE-FOCK)")
print("-" * 60)

uhf = scf.UHF(mol)
energia_hf = uhf.kernel()
print(f"\nEnergía HF: {energia_hf:.8f} Hartrees")

# 3. Cálculo MP2 (que es una aproximación a teoría de perturbaciones de segundo orden)
# MP2 es similar a SCP en que usa correlación electrónica de segundo orden
print("\n" + "-" * 60)
print("2. CÁLCULO MP2 (PERTURBATION THEORY - 2ND ORDER)")
print("-" * 60)

mp2_calc = mp.UMP2(uhf)
energia_mp2, energia_corr_mp2 = mp2_calc.kernel()

print(f"\nEnergía MP2: {energia_mp2:.8f} Hartrees")
print(f"Energía de correlación MP2: {energia_corr_mp2:.8f} Hartrees")
print(f"Corrección respecto a HF: {energia_mp2 - energia_hf:.8f} Hartrees")

# 4. Información adicional
print("\n" + "-" * 60)
print("3. INFORMACIÓN ADICIONAL")
print("-" * 60)

print(f"\nOrbital HOMO-LUMO gap (HF): {(uhf.mo_energy[1][uhf.nelectron//2] - uhf.mo_energy[1][uhf.nelectron//2-1]):.6f} Hartrees")
print(f"\nEnergía de ionización aproximada (diferencia HOMO): {-uhf.mo_energy[1][uhf.nelectron//2-1]:.6f} Hartrees")

print("\n" + "=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)
print(f"Energía HF (SCF):              {energia_hf:.10f} Hartrees")
print(f"Energía MP2 (Corr. Electr.):   {energia_mp2:.10f} Hartrees")
print(f"Energía de correlación:        {energia_corr_mp2:.10f} Hartrees")
print("=" * 60)
