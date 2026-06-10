import numpy as np
import pennylane as qp
import jax
import optax
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)
#Definimos tanto la molécual como el Hamiltoniano
symbols = ["Li","H"] 
coordinates = np.array([[0.0,0.0,0.0],[0.0,0.0,1.595]])                   
molecule = qp.qchem.Molecule(
    symbols,
    coordinates,
    basis_name="sto-3g"
)
H, qubits = qp.qchem.molecular_hamiltonian(molecule)
#print("Number of qubits = ", qubits)
#print("The Hamiltonian is ", H)

dev = qp.device("lightning.qubit", wires=qubits) #crea el dispositivo cuántico simulado donde PennyLane va a ejecutar el circuito.

#preparamos el estado hf (el de menor energia: ket(1100))     1: estado ocupado,    0: estado vacio
electrons = 4
singles, doubles = qp.qchem.excitations(electrons, qubits)
num_params = len(singles) + len(doubles)
hf = qp.qchem.hf_state(electrons, qubits)
#print(hf)
#se define el circuito variacional                             

@qp.qnode(dev, interface="jax")
def circuit(param, wires):
    #qp.BasisState(hf, wires=wires)

    qp.templates.AllSinglesDoubles(
    weights=param,
    wires=range(qubits),
    hf_state=hf,
    singles=singles,
    doubles=doubles
    )

    return qp.expval(H)

def cost_fn(param):
    return circuit(param, wires=range(qubits)) #se busca valor min. de energía

#optimizador
max_iterations = 300
conv_tol = 1e-06

opt = optax.sgd(learning_rate=0.3)

######################

theta = jax.numpy.zeros(num_params)

# store the values of the cost function
energy = [cost_fn(theta)]

# store the values of the circuit parameter
angle = [theta]

opt_state = opt.init(theta)

for n in range(max_iterations):

    gradient = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)

    angle.append(theta)
    energy.append(cost_fn(theta))

    conv = np.abs(energy[-1] - energy[-2])

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha") # esto va chivandote cual es el valor de la energia

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print("\nOptimal values of the circuit parameters:")
for i, value in enumerate(angle[-1]):
    print(f"theta[{i}] = {value:.4f}")

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Full configuration interaction (FCI) energy computed classically

mol = gto.M(
    atom='''
    Li 0 0 0
    H  0 0 1.595
    ''',
    basis='sto-3g'
)

mf = scf.RHF(mol)
mf.kernel()

cisolver = fci.FCI(mol, mf.mo_coeff)
E_fci, ci = cisolver.kernel()

print("FCI",E_fci)

# Add energy plot on column 1
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 2), energy, "go", ls="dashed")
ax1.plot(range(n + 2), np.full(n + 2, E_fci), color="red")
ax1.set_xlabel("Optimization step", fontsize=13)
ax1.set_ylabel("Energy (Hartree)", fontsize=13)
ax1.text(0.5, -1.1176, r"$E_\mathrm{HF}$", fontsize=15)
ax1.text(0, -1.1357, r"$E_\mathrm{FCI}$", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add angle plot on column 2
angle_array = np.array(angle)

ax2 = fig.add_subplot(122)

for i in range(4):
    ax2.plot(range(n + 2), angle_array[:, i], "o", ls="dashed", label=f"$\\theta_{i}$")

ax2.set_xlabel("Optimization step", fontsize=13)
ax2.set_ylabel("Gate parameters $\\theta_i$ (rad)", fontsize=13)
ax2.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.show()

