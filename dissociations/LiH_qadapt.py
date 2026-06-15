from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from scipy.optimize import minimize
from itertools import product

from pyscf import fci

import matplotlib.pyplot as plt
import numpy as np


def construir_hamiltoniano_LiH(R):
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {R}",
        basis="sto3g",
        charge=0,
        spin=0,
    )

    problem = driver.run()

    mol = driver._mol
    mf = driver._calc

    transformer = ActiveSpaceTransformer(
        num_electrons=2,
        num_spatial_orbitals=2,
    )

    problem_reduced = transformer.transform(problem)

    hamiltoniano_fermionico = problem_reduced.hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    hamiltoniano_qubit = mapper.map(hamiltoniano_fermionico)

    return hamiltoniano_qubit, problem_reduced, mol, mf


def fci_in_active_space(H):
    H_matrix = H.to_matrix()
    eigvals = np.linalg.eigvalsh(H_matrix)
    return np.min(eigvals).real


def generar_pool_pauli(n_qubits):
    pool = []
    paulis = ["I", "X", "Y"]

    for ops in product(paulis, repeat=n_qubits):
        pauli_string = "".join(ops)

        if pauli_string == "I" * n_qubits:
            continue

        if "X" in pauli_string or "Y" in pauli_string:
            pool.append(SparsePauliOp.from_list([(pauli_string, 1.0)]))

    return pool


def initial_state(n_qubits):
    qc = QuantumCircuit(n_qubits)

    qc.x(0)
    qc.x(1)

    return qc


def add_pauli_evolution(qc, pauli_string, theta):
    active = []

    for i, p in enumerate(pauli_string):
        if p == "X":
            qc.h(i)
            active.append(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)
            active.append(i)

    if len(active) == 0:
        return

    for i in active[:-1]:
        qc.cx(i, active[-1])

    qc.rz(2 * theta, active[-1])

    for i in reversed(active[:-1]):
        qc.cx(i, active[-1])

    for i, p in enumerate(pauli_string):
        if p == "X":
            qc.h(i)
        elif p == "Y":
            qc.h(i)
            qc.s(i)


def build_ansatz(n_qubits, selected_ops, params):
    qc = initial_state(n_qubits)

    for op, theta in zip(selected_ops, params):
        pauli_string = op.paulis[0].to_label()
        add_pauli_evolution(qc, pauli_string, theta)

    return qc


def energy(params, H, selected_ops):
    n_qubits = H.num_qubits
    qc = build_ansatz(n_qubits, selected_ops, params)
    psi = Statevector.from_instruction(qc)

    return np.real(psi.expectation_value(H))


def gradient_for_operator(H, selected_ops, params, op, delta=1e-4):
    ops_plus = selected_ops + [op]

    params_plus = np.append(params, delta)
    params_minus = np.append(params, -delta)

    e_plus = energy(params_plus, H, ops_plus)
    e_minus = energy(params_minus, H, ops_plus)

    return (e_plus - e_minus) / (2 * delta)


def qadapt_vqe(H, n_qubits, max_iter=10, gradient_tol=1e-4):
    selected_ops = []
    params = np.array([])

    pool = generar_pool_pauli(n_qubits)

    for iteration in range(max_iter):
        gradients = []

        for op in pool:
            grad = gradient_for_operator(H, selected_ops, params, op)
            gradients.append(abs(grad))

        best_index = int(np.argmax(gradients))
        best_gradient = gradients[best_index]
        best_op = pool[best_index]

        print(f"Iteración {iteration}")
        print(f"Mayor gradiente = {best_gradient:.6e}")
        print(f"Operador elegido = {best_op.paulis[0].to_label()}")

        if best_gradient < gradient_tol:
            print("Convergencia alcanzada.")
            break

        selected_ops.append(best_op)
        params = np.append(params, 0.0)

        result = minimize(
            energy,
            params,
            args=(H, selected_ops),
            method="BFGS"
        )

        params = result.x

        print(f"Energía optimizada = {result.fun:.8f} Ha")
        print()

    return energy(params, H, selected_ops)


distancias = np.linspace(0.4, 3.5, 25)

energias_qadapt = []
energias_fci = []

for R in distancias:
    H_qubit, problem, mol, mf = construir_hamiltoniano_LiH(R)
    n_qubits = H_qubit.num_qubits

    E_qadapt = qadapt_vqe(
        H=H_qubit,
        n_qubits=n_qubits,
        max_iter=10,
        gradient_tol=1e-4
    )

    E_fci = fci_in_active_space(H_qubit)

    energias_qadapt.append(E_qadapt)
    energias_fci.append(E_fci)

    print(f"R = {R:.2f} Å")
    print(f"Q-ADAPT-VQE = {E_qadapt:.8f} Ha")
    print(f"FCI-in HF   = {E_fci:.8f} Ha")
    print("-" * 50)


plt.plot(distancias, energias_qadapt, "o-", label="Q-ADAPT-VQE")
plt.plot(distancias, energias_fci, "s--", label="FCI-in HF PySCF")

plt.xlabel("Distancia Li-H (Å)")
plt.ylabel("Energía (Ha)")
plt.legend()
plt.grid()
plt.show()