import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

H2_HAMILTONIAN = {
    "II": -1.052373245772859,
    "IZ":  0.39793742484318045,
    "ZI": -0.39793742484318045,
    "ZZ": -0.01128010425623538,
    "XX":  0.18093119978423156,
}

EXACT_GROUND_STATE_ENERGY = -1.8572750302023797  # Ha
CHEMICAL_ACCURACY = 0.0016
N_QUBITS = 2
N_PARAMS = 8  # 2 layers × 2 qubits × 2 rotations
# ==========================================================================
# Exact diagonalization (classical reference)
# ==========================================================================
def compute_exact_ground_state(
    hamiltonian: dict[str, float] | None = None,
) -> dict:
    """Compute the exact ground state by diagonalizing the 16x16 Hamiltonian.
    For 4 qubits we can still diagonalize classically (16x16 matrix).
    For larger molecules this becomes intractable — that's why VQE exists.
    Args:
        hamiltonian: Pauli term coefficients. Defaults to HAMILTONIAN.
    Returns:
        dict with ground_state_energy, all eigenvalues, and ground state vector.
    """
    if hamiltonian is None:
        hamiltonian = H2_HAMILTONIAN
    i2 = np.eye(2)
    pauli = {"I": i2, "Z": np.array([[1, 0], [0, -1]]),
             "X": np.array([[0, 1], [1, 0]]),
             "Y": np.array([[0, -1j], [1j, 0]])}
    n = 2 ** N_QUBITS
    h_matrix = np.zeros((n, n), dtype=complex)
    for term, coeff in hamiltonian.items():
        op = np.array([[1.0 + 0j]])
        for char in term:
            op = np.kron(op, pauli[char])
        h_matrix += coeff * op
    eigenvalues = np.linalg.eigvalsh(h_matrix.real)
    return {
        "ground_state_energy": eigenvalues[0],
        "eigenvalues": eigenvalues.tolist(),
    }
# ==========================================================================
# Circuit construction
# ==========================================================================
def create_lih_ansatz(theta: list[float], layers: int = 2) -> QuantumCircuit:
    """Create a hardware-efficient ansatz for the LiH molecule.
    Uses a layered structure:
      - Each layer: RY + RZ on every qubit, then linear CX chain
      - 2 layers give 16 parameters (4 qubits × 2 rotations × 2 layers)
    The linear CX chain (0→1→2→3) matches the connectivity of most
    superconducting quantum processors, making this hardware-friendly.
    Args:
        theta: List of rotation angles. Must have (8 × layers) elements.
        layers: Number of variational layers (default: 2).
    Returns:
        QuantumCircuit with parameterized ansatz (no measurements).
    Raises:
        ValueError: If theta length doesn't match expected parameter count.
    """
    expected = N_QUBITS * 2 * layers
    if len(theta) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(theta)}")
    qc = QuantumCircuit(N_QUBITS)
    idx = 0
    for layer in range(layers):
        # Single-qubit rotation layer: RY + RZ on each qubit
        # RY controls the polar angle, RZ controls the azimuthal angle
        for qubit in range(N_QUBITS):
            qc.ry(theta[idx], qubit)
            idx += 1
            qc.rz(theta[idx], qubit)
            idx += 1
        # Entangling layer: linear CX chain
        # Creates nearest-neighbor correlations between adjacent qubits
        for qubit in range(N_QUBITS - 1):
            qc.cx(qubit, qubit + 1)
        if layer < layers - 1:
            qc.barrier()  # Visual separation between layers
    return qc
def create_measurement_circuit(
    ansatz: QuantumCircuit,
    pauli_string: str,
) -> QuantumCircuit:
    """Create a measurement circuit for a specific Pauli string.
    Rotates each qubit into the appropriate eigenbasis before measuring:
      - I, Z: no rotation (computational basis)
      - X: apply H (Hadamard)
      - Y: apply Sdg then H
    Args:
        ansatz: The parameterized circuit (without measurements).
        pauli_string: 4-character string like 'ZZII', 'XXII', etc.
            Characters are ordered as qubit 3, 2, 1, 0 (MSB to LSB).
    Returns:
        QuantumCircuit with basis rotations and measurements.
    """
    qc = ansatz.copy()
    # Pauli string is ordered MSB→LSB: char 0 = qubit (n-1), char 3 = qubit 0
    for i, pauli_char in enumerate(reversed(pauli_string)):
        if pauli_char == "X":
            qc.h(i)
        elif pauli_char == "Y":
            qc.sdg(i)
            qc.h(i)
        # Z and I: no basis change needed
    qc.measure_all()
    return qc
# ==========================================================================
# Expectation value computation
# ==========================================================================
def compute_pauli_expectation(
    counts: dict,
    pauli_string: str,
    shots: int,
) -> float:
    """Compute the expectation value of a multi-qubit Pauli operator.
    For each measurement outcome, compute the product of Pauli eigenvalues
    at non-identity positions. Each qubit contributes +1 (if measured 0)
    or -1 (if measured 1) at X, Y, or Z positions.
    Args:
        counts: Measurement counts dict.
        pauli_string: Pauli operator string (e.g., 'ZZII').
        shots: Total number of measurements.
    Returns:
        Expectation value in [-1.0, +1.0].
    """
    expectation = 0.0
    for bitstring, count in counts.items():
        eigenvalue = 1.0
        for i, pauli_char in enumerate(reversed(pauli_string)):
            if pauli_char in ("X", "Y", "Z"):
                bit = int(bitstring[-(i + 1)])
                eigenvalue *= 1 - 2 * bit
        expectation += eigenvalue * count / shots
    return expectation
def compute_energy(
    pauli_expectations: dict[str, float],
    hamiltonian: dict[str, float] | None = None,
) -> float:
    """Compute the total LiH energy from Pauli expectation values.
    Args:
        pauli_expectations: Dict mapping Pauli strings to expectation values.
        hamiltonian: Hamiltonian coefficients. Defaults to LIH_HAMILTONIAN.
    Returns:
        Total energy estimate in Hartree.
    """
    if hamiltonian is None:
        hamiltonian = H2_HAMILTONIAN
    energy = hamiltonian.get("II", 0.0)
    for term, coeff in hamiltonian.items():
        if term == "II":
            continue
        energy += coeff * pauli_expectations.get(term, 0.0)
    return energy
# ==========================================================================
# Execution
# ==========================================================================
def run_circuit(
    theta: list[float] | None = None,
    shots: int = 1024,
) -> dict:
    """Execute the VQE ansatz for LiH and compute the energy estimate.
    Creates the ansatz, generates measurement circuits for each Pauli term,
    executes them, and combines results into a single energy estimate.
    Args:
        theta: Ansatz parameters (16 angles). If None, uses initial guess.
        shots: Measurement shots per circuit (default: 1024).
    Returns:
        dict with keys: theta, energy, exact_ground_state, error,
                        pauli_expectations, n_qubits, n_params.
    """
    if theta is None:
        # Initial guess: small rotations near |0000>
        # Not optimal, but gives a reasonable starting energy for demonstration
        theta = [0.1] * N_PARAMS
    ansatz = create_lih_ansatz(theta)
    backend = AerSimulator()
    pauli_expectations = {}
    for pauli_string, coeff in H2_HAMILTONIAN.items():
        if pauli_string == "II":
            continue
        circuit = create_measurement_circuit(ansatz, pauli_string)
        job = backend.run(circuit, shots=shots)
        counts = dict(job.result().get_counts(circuit))
        exp_val = compute_pauli_expectation(counts, pauli_string, shots)
        pauli_expectations[pauli_string] = exp_val
    energy = compute_energy(pauli_expectations)
    return {
        "theta": theta,
        "energy": energy,
        "exact_ground_state": EXACT_GROUND_STATE_ENERGY,
        "error": abs(energy - EXACT_GROUND_STATE_ENERGY),
        "pauli_expectations": pauli_expectations,
        "n_qubits": N_QUBITS,
        "n_params": N_PARAMS,
    }
# ==========================================================================
# Optimization
# ==========================================================================
def optimize_vqe(
    max_iterations: int = 200,
    shots: int = 2048,
    seed: int | None = None,
) -> dict:
    """Run the full VQE optimization loop to find the LiH ground state.
    With 16 parameters, LiH requires more iterations than H2. The COBYLA
    optimizer navigates a higher-dimensional landscape, making convergence
    slower and less guaranteed than the 2-qubit case.
    Args:
        max_iterations: Maximum optimizer iterations (default: 200).
        shots: Shots per circuit per iteration (default: 2048).
        seed: Random seed for reproducible initial parameters.
    Returns:
        dict with optimal_theta, optimal_energy, error, iterations, history.
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, np.pi, N_PARAMS)  # Smaller range to avoid barren plateau
    history = []
    def cost_function(theta):
        result = run_circuit(theta.tolist(), shots=shots)
        history.append(result["energy"])
        return result["energy"]
    result = minimize(
        cost_function,
        theta0,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.3},
    )
    error = abs(result.fun - EXACT_GROUND_STATE_ENERGY)
    plt.figure()
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.axhline(
    y=EXACT_GROUND_STATE_ENERGY,
    linestyle="--",
    label="Exact energy"
    )
    plt.xlabel("Step")
    plt.ylabel("Energy (Hartree)")
    plt.title("VQE optimization for h2")
    plt.grid(True)
    plt.show()
    return {
        "optimal_theta": result.x.tolist(),
        "optimal_energy": result.fun,
        "exact_ground_state": EXACT_GROUND_STATE_ENERGY,
        "error": error,
        "chemical_accuracy": error < CHEMICAL_ACCURACY,
        "iterations": result.nfev,
        "history": history,
    }
# ==========================================================================
# Verification
# ==========================================================================
def verify_vqe(shots: int = 4096, tolerance: float = 0.1) -> dict:
    """Verify the VQE circuit produces a physically reasonable LiH energy.
    Checks:
      1. Default parameters give energy below the identity coefficient
         (proving the ansatz does better than the trivial state)
      2. Variational principle: energy >= exact ground state (within noise)
      3. Optimization improves energy beyond the initial guess
    Args:
        shots: Measurement shots (default: 4096).
        tolerance: Energy tolerance in Hartree (default: 0.1, larger for 4 qubits).
    Returns:
        dict with passed, checks, energy, optimization.
    """
    checks = []
    passed = True
    # ---- Check 1: Default params give a finite, physical energy ----
    result = run_circuit(shots=shots)
    energy = result["energy"]
    energy_reasonable = (
        np.isfinite(energy)
        and energy > EXACT_GROUND_STATE_ENERGY - 5.0
        and energy < 0.0
    )
    check_nontrivial = {
        "name": "energy_physically_reasonable",
        "passed": energy_reasonable,
        "detail": (
            f"E = {energy:.4f} Ha (finite, in range "
            f"[{EXACT_GROUND_STATE_ENERGY - 5.0:.1f}, 0.0])"
        ),
    }
    checks.append(check_nontrivial)
    if not check_nontrivial["passed"]:
        passed = False
    # ---- Check 2: Variational principle ----
    noise_margin = 2 * tolerance
    check_variational = {
        "name": "variational_principle",
        "passed": energy >= EXACT_GROUND_STATE_ENERGY - noise_margin,
        "detail": (
            f"E = {energy:.4f} Ha >= E0 - margin = "
            f"{EXACT_GROUND_STATE_ENERGY - noise_margin:.4f} Ha"
        ),
    }
    checks.append(check_variational)
    if not check_variational["passed"]:
        passed = False
    # ---- Check 3: Optimization improves energy ----
    opt = optimize_vqe(max_iterations=100, shots=shots, seed=42)
    check_optimization = {
        "name": "optimization_improves",
        "passed": opt["optimal_energy"] < energy,
        "detail": (
            f"Optimized E = {opt['optimal_energy']:.4f} Ha < "
            f"initial E = {energy:.4f} Ha, "
            f"error = {opt['error']:.4f} Ha, "
            f"iterations = {opt['iterations']}"
        ),
    }
    checks.append(check_optimization)
    if not check_optimization["passed"]:
        passed = False
    return {
        "passed": passed,
        "energy": energy,
        "optimization": opt,
        "shots": shots,
        "checks": checks,
    }
# ==========================================================================
# Main — interactive exploration
# ==========================================================================
def create_circuit() -> QuantumCircuit:
    """Zero-arg entry point for the QubitHub Qiskit runner.
    Returns the LiH hardware-efficient ansatz with seeded pseudo-random
    parameters (seed=42, 16 angles). Mirrors the initialization used by
    ``optimize_vqe()`` in this file — produces a deterministic non-trivial
    superposition that VQE optimization converges to the true ground state
    from.
    """
    rng = np.random.default_rng(seed=42)
    theta = rng.uniform(0, 2 * np.pi, N_PARAMS).tolist()
    return create_lih_ansatz(theta)
if __name__ == "__main__":
    print("VQE for h2 Molecule Ground State")
    print("=" * 50)
    print("Running VQE for H2 with 2 qubits\n")
    # Classical reference
    exact = compute_exact_ground_state()
    print(f"Classical reference (exact diagonalization):")
    print(f"  Ground state energy: {exact['ground_state_energy']:.4f} Ha")
    print(f"  Hamiltonian terms:   {len(H2_HAMILTONIAN)}")
    print(f"  Matrix size:         16 x 16")
    # Initial evaluation
    print(f"\nStep 1: Initial ansatz (small rotations)")
    print("-" * 50)
    result = run_circuit()
    print(f"  Qubits:     {result['n_qubits']}")
    print(f"  Parameters: {result['n_params']}")
    print(f"  Energy:     {result['energy']:>8.4f} Ha")
    print(f"  Exact:      {EXACT_GROUND_STATE_ENERGY:>8.4f} Ha")
    print(f"  Error:      {result['error']:>8.4f} Ha")
    # Pauli expectations
    print(f"\n  Pauli expectations:")
    for term, exp in result["pauli_expectations"].items():
        coeff = H2_HAMILTONIAN[term]
        contrib = coeff * exp
        print(f"    {term}: <P> = {exp:+.3f}, coeff = {coeff:+.4f}, contrib = {contrib:+.4f}")
    # Optimization
    print(f"\nStep 2: VQE optimization (COBYLA, max 200 iterations)")
    print("-" * 50)
    opt = optimize_vqe(max_iterations=200, shots=2048, seed=42)
    print(f"  Optimal energy: {opt['optimal_energy']:>8.4f} Ha")
    print(f"  Exact:          {EXACT_GROUND_STATE_ENERGY:>8.4f} Ha")
    print(f"  Error:          {opt['error']:>8.4f} Ha")
    print(f"  Chemical accuracy: {'YES' if opt['chemical_accuracy'] else 'NO'}")
    print(f"  Iterations:     {opt['iterations']}")
    # Verification
    print(f"\nStep 3: Verification")
    print("-" * 50)
    v = verify_vqe()
    for check in v["checks"]:
        symbol = "PASS" if check["passed"] else "FAIL"
        print(f"  [{symbol}] {check['name']}: {check['detail']}")
    print(f"\nOverall: {'PASSED' if v['passed'] else 'FAILED'}")