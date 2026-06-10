#programa que usa AerSimulator para simular Hardwar, conociendo H aplica VQE
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from scipy.optimize import minimize
hamiltonian = SparsePauliOp.from_list(
    [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
)

A = np.array(hamiltonian)
eigenvalues, eigenvectors = np.linalg.eigh(A)
print("The ground state energy is ", min(eigenvalues))

# Pre-defined ansatz circuit and operator class for Hamiltonian


# Note that it is more common to place initial 'h' gates outside the ansatz. Here we specifically wanted this layer structure.
ansatz = efficient_su2(
    hamiltonian.num_qubits, su2_gates=["h", "rz", "y"], entanglement="circular", reps=1
)

num_params = ansatz.num_parameters
print("This circuit has ", num_params, "parameters")

ansatz.decompose().draw("mpl", style="iqp")
# runtime imports


# To run on hardware, select the backend with the fewest number of jobs in the queue
#service = QiskitRuntimeService()
#backend = service.least_busy(operational=True, simulator=False)
backend=AerSimulator()

print(backend)


target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)

ansatz_isa = pm.run(ansatz)

ansatz_isa.draw(output="mpl", idle_wires=False, style="iqp")
hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from Estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}
x0 = 2 * np.pi * np.random.random(num_params)
# This required 13 min, 20 s QPU time on an Eagle processor, 28 min total time.
estimator = Estimator(mode=backend)
estimator.options.default_shots = 10000

res = minimize(
    cost_func,
    x0,
    args=(ansatz_isa, hamiltonian_isa, estimator),
    method="COBYLA",
    options={"maxiter": 50},
)
steps = range(1, len(cost_history_dict["cost_history"]) + 1)
energies = cost_history_dict["cost_history"]

plt.figure()
plt.plot(steps, energies, marker="o", label="VQE energy")
plt.axhline(
    y=min(eigenvalues),
    linestyle="--",
    label="Exact energy"
)
plt.xlabel("Step")
plt.ylabel("Energy (Hartree)")
plt.title("VQE optimization")
plt.grid(True)
plt.legend()
plt.show()