#programa que usa AerSimulator para simular Hardwar, conociendo H aplica VQE
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.circuit.library import efficient_su2, real_amplitudes, TwoLocal
hamiltonian = SparsePauliOp.from_list(
    [
        ("IIII", -7.2062),
        ("ZZII", 0.1722),
        ("IIZZ", 0.1209),
        ("ZIZI", 0.1659),
        ("IZIZ", 0.1659),
        ("XXII", 0.0454),
        ("YYII", 0.0454),
        ("IIXX", 0.0505),
        ("IIYY", 0.0505),
    ]
)

A = np.array(hamiltonian)
eigenvalues, eigenvectors = np.linalg.eigh(A)
exact_energy = min(eigenvalues)
ansatz_list = {
    "EfficientSU2": efficient_su2(
        hamiltonian.num_qubits,
        su2_gates=["h", "rz", "y"],
        entanglement="circular",
        reps=1,
    ),

    "RealAmplitudes": real_amplitudes(
        hamiltonian.num_qubits,
        entanglement="linear",
        reps=2,
    ),

    "TwoLocal_RY_RZ": TwoLocal(
        hamiltonian.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2,
    ),
}


# runtime imports


# To run on hardware, select the backend with the fewest number of jobs in the queue
#service = QiskitRuntimeService()
#backend = service.least_busy(operational=True, simulator=False)
backend=AerSimulator()


target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
estimator = Estimator(mode=backend)
estimator.options.default_shots = 10000


def cost_func(params, ansatz, hamiltonian, estimator, cost_history_dict):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)

    print(f"Iters: {cost_history_dict['iters']} | Energy: {energy}")

    return energy


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}


results = {}

for name, ansatz in ansatz_list.items():
    print("\nRunning ansatz:", name)

    num_params = ansatz.num_parameters
    x0 = 2 * np.pi * np.random.random(num_params)

    ansatz_isa = pm.run(ansatz)
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    res = minimize(
        cost_func,
        x0,
        args=(ansatz_isa, hamiltonian_isa, estimator, cost_history_dict),
        method="COBYLA",
        options={"maxiter": 50},
    )

    results[name] = {
        "energy": res.fun,
        "params": res.x,
        "history": cost_history_dict["cost_history"],
    }


plt.figure()

for name, data in results.items():
    plt.plot(
        range(1, len(data["history"]) + 1),
        data["history"],
        marker="o",
        label=name,
    )

plt.axhline(
    y=exact_energy,
    linestyle="--",
    label=f"Exact = {exact_energy:.6f} Ha",
)

plt.xlabel("Step")
plt.ylabel("Energy (Hartree)")
plt.title("VQE comparison with different ansatz")
plt.grid(True)
plt.legend()
plt.show()
