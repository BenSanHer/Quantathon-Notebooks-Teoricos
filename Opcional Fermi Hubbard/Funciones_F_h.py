"""Funciones para la simulación Trotter de Fermi-Hubbard en una cadena 1D.

Estas funciones fueron extraídas de ``Trotterizacion Fermi Hubbard.ipynb``
para que el notebook contenga únicamente la explicación y los experimentos.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


__all__ = [
    "r_xzx",
    "r_yzy",
    "aplicar_hopping_enlace",
    "aplicar_interaccion_sitio",
    "H_FH_trott1_o",
    "ocupaciones_modos",
    "densidad_por_sitio",
    "doble_ocupacion_por_sitio",
    "doble_ocupacion_promedio",
    "evolucion_FH",
    "estado_inicial",
    "hamiltoniano_FH_exacto",
    "evolucion_FH_exacta",
    "observables_estado_exacto",
    "barrido_estado_fundamental",
    "estilo_graficas",
    "graficar_densidades",
    "ansatz_rotaciones_FH",
    "observables_estado_pennylane",
    "vqe_FH_pennylane",
    "barrido_vqe_FH",
]


def r_xzx(
    qc: QuantumCircuit,
    theta: float,
    q0: int,
    q1: int,
    q2: int,
) -> None:
    """Implementa ``exp(-i theta X_q0 Z_q1 X_q2 / 2)``."""
    qc.h(q0)
    qc.h(q2)
    qc.cx(q0, q2)
    qc.cx(q1, q2)
    qc.rz(theta, q2)
    qc.cx(q1, q2)
    qc.cx(q0, q2)
    qc.h(q2)
    qc.h(q0)


def r_yzy(
    qc: QuantumCircuit,
    theta: float,
    q0: int,
    q1: int,
    q2: int,
) -> None:
    """Implementa ``exp(-i theta Y_q0 Z_q1 Y_q2 / 2)``."""
    qc.sdg(q0)
    qc.h(q0)
    qc.sdg(q2)
    qc.h(q2)
    qc.cx(q0, q2)
    qc.cx(q1, q2)
    qc.rz(theta, q2)
    qc.cx(q1, q2)
    qc.cx(q0, q2)
    qc.h(q2)
    qc.s(q2)
    qc.h(q0)
    qc.s(q0)


def aplicar_hopping_enlace(
    qc: QuantumCircuit,
    hop: float,
    dt: float,
    i: int,
) -> None:
    """Aplica la evolución del hopping entre los sitios ``i`` e ``i+1``."""
    theta = -hop * dt
    q_up_i = 2 * i
    q_down_i = 2 * i + 1
    q_up_j = 2 * i + 2
    q_down_j = 2 * i + 3

    r_xzx(qc, theta, q_up_i, q_down_i, q_up_j)
    r_yzy(qc, theta, q_up_i, q_down_i, q_up_j)
    r_xzx(qc, theta, q_down_i, q_up_j, q_down_j)
    r_yzy(qc, theta, q_down_i, q_up_j, q_down_j)


def aplicar_interaccion_sitio(
    qc: QuantumCircuit,
    U: float,
    dt: float,
    i: int,
) -> None:
    """Aplica ``U n_up n_down`` omitiendo la fase global."""
    q_up = 2 * i
    q_down = 2 * i + 1
    qc.rz(-U * dt / 2, q_up)
    qc.rz(-U * dt / 2, q_down)
    qc.rzz(U * dt / 2, q_up, q_down)


def H_FH_trott1_o(
    hop: float,
    U: float,
    n_q: int,
    r: int,
    t: float,
    occupied_modes: Sequence[int] | None = None,
) -> Statevector:
    """Evoluciona una cadena abierta con Trotter de primer orden."""
    if n_q % 2 != 0:
        raise ValueError("n_q debe ser par: cada sitio utiliza dos qubits.")
    if r <= 0:
        raise ValueError("r debe ser positivo.")

    n_sites = n_q // 2
    dt = t / r
    qc = QuantumCircuit(n_q)

    if occupied_modes is not None:
        for q in occupied_modes:
            if q < 0 or q >= n_q:
                raise ValueError(f"El modo {q} no existe.")
            qc.x(q)

    for _ in range(r):
        for i in range(0, n_sites - 1, 2):
            aplicar_hopping_enlace(qc, hop, dt, i)
        for i in range(1, n_sites - 1, 2):
            aplicar_hopping_enlace(qc, hop, dt, i)
        for i in range(n_sites):
            aplicar_interaccion_sitio(qc, U, dt, i)

    return Statevector.from_instruction(qc)


def ocupaciones_modos(state: Statevector) -> np.ndarray:
    """Calcula ``<n_q>`` para cada modo fermiónico."""
    probabilities = state.probabilities()
    occupations = np.zeros(state.num_qubits)

    for basis_index, probability in enumerate(probabilities):
        for q in range(state.num_qubits):
            occupation = (basis_index >> q) & 1
            occupations[q] += occupation * probability

    return occupations


def densidad_por_sitio(state: Statevector) -> np.ndarray:
    """Devuelve ``<n_up+n_down>`` para cada sitio físico."""
    occupations = ocupaciones_modos(state)
    n_sites = state.num_qubits // 2
    densities = np.zeros(n_sites)

    for i in range(n_sites):
        q_up = 2 * i
        q_down = 2 * i + 1
        densities[i] = occupations[q_up] + occupations[q_down]

    return densities


def doble_ocupacion_por_sitio(state: Statevector) -> np.ndarray:
    """Calcula ``<n_up n_down>`` para cada sitio físico."""
    probabilities = state.probabilities()
    n_sites = state.num_qubits // 2
    double_occupations = np.zeros(n_sites)

    for basis_index, probability in enumerate(probabilities):
        for i in range(n_sites):
            q_up = 2 * i
            q_down = 2 * i + 1
            n_up = (basis_index >> q_up) & 1
            n_down = (basis_index >> q_down) & 1
            double_occupations[i] += n_up * n_down * probability

    return double_occupations


def doble_ocupacion_promedio(state: Statevector) -> float:
    """Promedia la doble ocupación sobre todos los sitios."""
    return float(np.mean(doble_ocupacion_por_sitio(state)))


def evolucion_FH(
    hop: float,
    U: float,
    n_q: int,
    r: int,
    tiempos: np.ndarray,
    occupied_modes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula densidad, doble ocupación y número total frente al tiempo."""
    densities_history = []
    double_history = []
    particle_number_history = []

    for time in tiempos:
        state = H_FH_trott1_o(
            hop=hop,
            U=U,
            n_q=n_q,
            r=r,
            t=float(time),
            occupied_modes=occupied_modes,
        )
        densities = densidad_por_sitio(state)
        densities_history.append(densities)
        double_history.append(doble_ocupacion_promedio(state))
        particle_number_history.append(np.sum(densities))

    return (
        np.asarray(densities_history),
        np.asarray(double_history),
        np.asarray(particle_number_history),
    )


def estado_inicial(n_q: int, occupied_modes: Sequence[int]) -> Statevector:
    """Construye el estado computacional definido por los modos ocupados."""
    if len(set(occupied_modes)) != len(occupied_modes):
        raise ValueError("occupied_modes no puede contener modos repetidos.")
    basis_index = 0
    for q in occupied_modes:
        if q < 0 or q >= n_q:
            raise ValueError(f"El modo {q} no existe.")
        basis_index |= 1 << q
    vector = np.zeros(2**n_q, dtype=complex)
    vector[basis_index] = 1.0
    return Statevector(vector)


def _signo_fermi(bitstring: int, mode: int) -> int:
    """Signo de Jordan-Wigner producido por los modos menores que ``mode``."""
    return -1 if (bitstring & ((1 << mode) - 1)).bit_count() % 2 else 1


def _aplicar_cdag_c(bitstring: int, destino: int, origen: int):
    """Aplica c†_destino c_origen a un estado de la base de ocupación."""
    if not (bitstring >> origen) & 1 or (bitstring >> destino) & 1:
        return None
    signo = _signo_fermi(bitstring, origen)
    intermedio = bitstring ^ (1 << origen)
    signo *= _signo_fermi(intermedio, destino)
    return intermedio | (1 << destino), signo


def _base_sector(n_sites: int, n_up: int, n_down: int) -> list[int]:
    """Base de Fock con números de partículas por espín fijos."""
    if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
        raise ValueError("n_up y n_down deben estar entre 0 y n_sites.")
    base = []
    for up_sites in combinations(range(n_sites), n_up):
        up_mask = sum(1 << (2 * i) for i in up_sites)
        for down_sites in combinations(range(n_sites), n_down):
            down_mask = sum(1 << (2 * i + 1) for i in down_sites)
            base.append(up_mask | down_mask)
    return base


def hamiltoniano_FH_exacto(
    n_sites: int,
    hop: float,
    U: float,
    n_up: int,
    n_down: int,
) -> tuple[np.ndarray, list[int]]:
    """Matriz exacta de Fermi-Hubbard 1D abierta en un sector (N↑, N↓)."""
    base = _base_sector(n_sites, n_up, n_down)
    indice = {state: i for i, state in enumerate(base)}
    H = np.zeros((len(base), len(base)), dtype=float)

    for column, bitstring in enumerate(base):
        dobles = sum(
            ((bitstring >> (2 * site)) & 1)
            * ((bitstring >> (2 * site + 1)) & 1)
            for site in range(n_sites)
        )
        H[column, column] = U * dobles

        for site in range(n_sites - 1):
            for spin in (0, 1):
                left = 2 * site + spin
                right = 2 * (site + 1) + spin
                for destino, origen in ((left, right), (right, left)):
                    resultado = _aplicar_cdag_c(bitstring, destino, origen)
                    if resultado is not None:
                        new_state, sign = resultado
                        H[indice[new_state], column] += -hop * sign

    return H, base


def observables_estado_exacto(
    amplitudes: np.ndarray,
    base: Sequence[int],
    n_sites: int,
) -> dict[str, np.ndarray | float]:
    """Densidad, doble ocupación y correlación de espín escalonada."""
    probabilities = np.abs(amplitudes) ** 2
    densities = np.zeros(n_sites)
    doubles = np.zeros(n_sites)
    spin_corr = np.zeros((n_sites, n_sites))

    for probability, bitstring in zip(probabilities, base):
        n_up = np.array([(bitstring >> (2 * i)) & 1 for i in range(n_sites)])
        n_down = np.array(
            [(bitstring >> (2 * i + 1)) & 1 for i in range(n_sites)]
        )
        density = n_up + n_down
        spin_z = 0.5 * (n_up - n_down)
        densities += probability * density
        doubles += probability * n_up * n_down
        spin_corr += probability * np.outer(spin_z, spin_z)

    staggered = sum(
        (-1) ** (i - j) * spin_corr[i, j]
        for i in range(n_sites)
        for j in range(n_sites)
    ) / n_sites
    return {
        "densidad": densities,
        "doble_ocupacion_sitios": doubles,
        "doble_ocupacion": float(np.mean(doubles)),
        "correlacion_spin": spin_corr,
        "estructura_spin_pi": float(staggered),
    }


def evolucion_FH_exacta(
    hop: float,
    U: float,
    n_sites: int,
    tiempos: Sequence[float],
    occupied_modes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolución exacta en el sector fijado por el estado inicial."""
    n_up = sum(q % 2 == 0 for q in occupied_modes)
    n_down = len(occupied_modes) - n_up
    H, base = hamiltoniano_FH_exacto(n_sites, hop, U, n_up, n_down)
    initial_bits = sum(1 << q for q in occupied_modes)
    if initial_bits not in base:
        raise ValueError("El estado inicial no pertenece al sector construido.")
    initial = np.zeros(len(base), dtype=complex)
    initial[base.index(initial_bits)] = 1.0
    energies, eigenvectors = np.linalg.eigh(H)
    coefficients = eigenvectors.conj().T @ initial

    densities_history = []
    double_history = []
    particle_history = []
    for time in tiempos:
        state = eigenvectors @ (np.exp(-1j * energies * time) * coefficients)
        obs = observables_estado_exacto(state, base, n_sites)
        densities_history.append(obs["densidad"])
        double_history.append(obs["doble_ocupacion"])
        particle_history.append(np.sum(obs["densidad"]))
    return (
        np.asarray(densities_history),
        np.asarray(double_history),
        np.asarray(particle_history),
    )


def _energia_fundamental(
    n_sites: int,
    hop: float,
    U: float,
    n_particles: int,
) -> float:
    """Menor energía entre todos los sectores de espín con N fijo."""
    sectors = (
        (n_up, n_particles - n_up)
        for n_up in range(n_sites + 1)
        if 0 <= n_particles - n_up <= n_sites
    )
    return min(
        np.linalg.eigvalsh(
            hamiltoniano_FH_exacto(n_sites, hop, U, n_up, n_down)[0]
        )[0]
        for n_up, n_down in sectors
    )


def barrido_estado_fundamental(
    valores_U: Sequence[float],
    n_sites: int = 4,
    hop: float = 1.0,
) -> dict[str, np.ndarray]:
    """Observables exactos a media ocupación para diagnosticar el régimen Mott."""
    if n_sites % 2:
        raise ValueError("Use un número par de sitios para media ocupación.")
    n_up = n_down = n_sites // 2
    doubles = []
    spin_structure = []
    charge_gaps = []
    energies = []

    for U in valores_U:
        H, base = hamiltoniano_FH_exacto(n_sites, hop, float(U), n_up, n_down)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        energies.append(eigenvalues[0])
        obs = observables_estado_exacto(eigenvectors[:, 0], base, n_sites)
        doubles.append(obs["doble_ocupacion"])
        spin_structure.append(obs["estructura_spin_pi"])
        e_minus = _energia_fundamental(
            n_sites, hop, float(U), n_sites - 1
        )
        e_plus = _energia_fundamental(
            n_sites, hop, float(U), n_sites + 1
        )
        charge_gaps.append(e_plus + e_minus - 2 * eigenvalues[0])

    return {
        "U": np.asarray(valores_U, dtype=float),
        "energia": np.asarray(energies),
        "doble_ocupacion": np.asarray(doubles),
        "brecha_carga": np.asarray(charge_gaps),
        "estructura_spin_pi": np.asarray(spin_structure),
    }


def estilo_graficas() -> None:
    """Estilo compacto y legible para las figuras del notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 4.8),
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def graficar_densidades(
    tiempos: Sequence[float],
    densities: np.ndarray,
    titulo: str,
    ax=None,
):
    """Grafica la densidad temporal de todos los sitios con sintaxis breve."""
    if ax is None:
        _, ax = plt.subplots()
    for site in range(densities.shape[1]):
        ax.plot(tiempos, densities[:, site], lw=2, label=f"Sitio {site}")
    ax.set(
        xlabel=r"Tiempo $J\tau$",
        ylabel=r"$\langle n_i\rangle$",
        title=titulo,
    )
    ax.legend(ncol=2)
    return ax


# =============================================================================
# VQE CON PENNYLANE
# =============================================================================


def _dependencias_vqe_FH():
    """Importa PennyLane únicamente cuando se solicita un cálculo VQE."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as error:
        raise ImportError(
            "El VQE de Fermi-Hubbard requiere PennyLane."
        ) from error
    return qml, pnp


def _matriz_hubbard_completa(
    n_sites: int,
    hop: float,
    U: float,
) -> np.ndarray:
    """Hamiltoniano físico en toda la base de Fock, en el orden de PennyLane."""
    n_qubits = 2 * n_sites
    dimension = 2**n_qubits
    H = np.zeros((dimension, dimension), dtype=complex)

    def indice_pennylane(bitstring: int) -> int:
        return sum(
            ((bitstring >> q) & 1) << (n_qubits - 1 - q)
            for q in range(n_qubits)
        )

    for bitstring in range(dimension):
        column = indice_pennylane(bitstring)
        dobles = sum(
            ((bitstring >> (2 * site)) & 1)
            * ((bitstring >> (2 * site + 1)) & 1)
            for site in range(n_sites)
        )
        H[column, column] = U * dobles

        for site in range(n_sites - 1):
            for spin in (0, 1):
                left = 2 * site + spin
                right = 2 * (site + 1) + spin
                for destino, origen in ((left, right), (right, left)):
                    resultado = _aplicar_cdag_c(bitstring, destino, origen)
                    if resultado is not None:
                        new_state, sign = resultado
                        row = indice_pennylane(new_state)
                        H[row, column] += -hop * sign
    return H


def _matriz_penalizacion_sector(
    n_sites: int,
    n_up_objetivo: int,
    n_down_objetivo: int,
) -> np.ndarray:
    """Matriz diagonal que vale cero sólo en el sector físico objetivo."""
    n_qubits = 2 * n_sites
    dimension = 2**n_qubits
    diagonal = np.zeros(dimension)

    for index in range(dimension):
        bits = [
            (index >> (n_qubits - 1 - q)) & 1
            for q in range(n_qubits)
        ]
        n_up = sum(bits[0::2])
        n_down = sum(bits[1::2])
        diagonal[index] = (
            (n_up - n_up_objetivo) ** 2
            + (n_down - n_down_objetivo) ** 2
        )
    return np.diag(diagonal)


def ansatz_rotaciones_FH(
    parametros,
    occupied_modes: Sequence[int],
    n_qubits: int,
    n_layers: int,
) -> None:
    """Ansatz hardware-efficient: Rot(φ, θ, ω) por qubit y anillos de CNOT."""
    qml, _ = _dependencias_vqe_FH()
    ocupacion = np.zeros(n_qubits, dtype=int)
    ocupacion[list(occupied_modes)] = 1
    qml.BasisState(ocupacion, wires=range(n_qubits))

    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.Rot(*parametros[layer, q], wires=q)
        if n_qubits == 2:
            qml.CNOT(wires=[0, 1])
        else:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])


def _estado_exacto_embebido(
    n_sites: int,
    hop: float,
    U: float,
    n_up: int,
    n_down: int,
) -> tuple[float, np.ndarray]:
    """Estado exacto del sector fijo embebido en los 2L qubits de PennyLane."""
    H_sector, base = hamiltoniano_FH_exacto(
        n_sites, hop, U, n_up, n_down
    )
    energies, vectors = np.linalg.eigh(H_sector)
    n_qubits = 2 * n_sites
    full_state = np.zeros(2**n_qubits, dtype=complex)
    for amplitude, bitstring in zip(vectors[:, 0], base):
        index = sum(
            ((bitstring >> q) & 1) << (n_qubits - 1 - q)
            for q in range(n_qubits)
        )
        full_state[index] = amplitude
    return float(energies[0]), full_state


def observables_estado_pennylane(
    state: np.ndarray,
    n_sites: int,
) -> dict[str, np.ndarray | float]:
    """Mide densidad, doublones y S(pi) en un estado completo de PennyLane."""
    n_qubits = 2 * n_sites
    state = np.asarray(state, dtype=complex)
    if state.shape != (2**n_qubits,):
        raise ValueError("El estado no tiene la dimensión esperada.")

    probabilities = np.abs(state) ** 2
    densities = np.zeros(n_sites)
    doubles = np.zeros(n_sites)
    spin_corr = np.zeros((n_sites, n_sites))

    for index, probability in enumerate(probabilities):
        occupations = np.array(
            [
                (index >> (n_qubits - 1 - q)) & 1
                for q in range(n_qubits)
            ]
        )
        n_up = occupations[0::2]
        n_down = occupations[1::2]
        density = n_up + n_down
        spin_z = 0.5 * (n_up - n_down)
        densities += probability * density
        doubles += probability * n_up * n_down
        spin_corr += probability * np.outer(spin_z, spin_z)

    staggered = sum(
        (-1) ** (i - j) * spin_corr[i, j]
        for i in range(n_sites)
        for j in range(n_sites)
    ) / n_sites
    return {
        "densidad": densities,
        "doble_ocupacion_sitios": doubles,
        "doble_ocupacion": float(np.mean(doubles)),
        "correlacion_spin": spin_corr,
        "estructura_spin_pi": float(staggered),
    }


def vqe_FH_pennylane(
    U: float,
    hop: float = 1.0,
    n_sites: int = 2,
    n_layers: int = 3,
    learning_rate: float = 0.06,
    max_steps: int = 350,
    tol: float = 1e-8,
    patience: int = 25,
    seed: int = 7,
    parametros_iniciales=None,
) -> dict:
    """VQE de Fermi-Hubbard a media ocupación usando PennyLane y Adam."""
    qml, pnp = _dependencias_vqe_FH()
    if n_sites % 2:
        raise ValueError("n_sites debe ser par para usar media ocupación balanceada.")
    if n_layers < 1:
        raise ValueError("n_layers debe ser positivo.")

    n_qubits = 2 * n_sites
    n_up = n_down = n_sites // 2
    occupied_modes = [
        mode
        for site in range(n_sites)
        for mode in ([2 * site] if site % 2 == 0 else [2 * site + 1])
    ]
    matriz_fisica = _matriz_hubbard_completa(n_sites, hop, U)
    matriz_sector = _matriz_penalizacion_sector(
        n_sites, n_up, n_down
    )
    escala_penalizacion = 4.0 * abs(hop) + abs(U) + 1.0
    matriz_objetivo = matriz_fisica + escala_penalizacion * matriz_sector

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)
    H_fisico = qml.Hermitian(matriz_fisica, wires=range(n_qubits))
    H_objetivo = qml.Hermitian(matriz_objetivo, wires=range(n_qubits))
    P_sector = qml.Hermitian(
        np.diag(np.diag(matriz_sector) == 0).astype(float),
        wires=range(n_qubits),
    )

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_objetivo(parametros):
        ansatz_rotaciones_FH(
            parametros, occupied_modes, n_qubits, n_layers
        )
        return qml.expval(H_objetivo)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_fisico(parametros):
        ansatz_rotaciones_FH(
            parametros, occupied_modes, n_qubits, n_layers
        )
        return qml.expval(H_fisico)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_sector(parametros):
        ansatz_rotaciones_FH(
            parametros, occupied_modes, n_qubits, n_layers
        )
        return qml.expval(P_sector)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_estado(parametros):
        ansatz_rotaciones_FH(
            parametros, occupied_modes, n_qubits, n_layers
        )
        return qml.state()

    if parametros_iniciales is None:
        rng = np.random.default_rng(seed)
        parametros = pnp.array(
            rng.normal(0.0, 0.08, size=(n_layers, n_qubits, 3)),
            requires_grad=True,
        )
    else:
        parametros = pnp.array(
            np.asarray(parametros_iniciales), requires_grad=True
        )

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    history = [float(circuito_objetivo(parametros))]
    stable_steps = 0
    for _ in range(max_steps):
        parametros = optimizer.step(circuito_objetivo, parametros)
        energy = float(circuito_objetivo(parametros))
        history.append(energy)
        difference = abs(history[-1] - history[-2])
        stable_steps = stable_steps + 1 if difference < tol else 0
        if stable_steps >= patience:
            break

    state_vqe = np.asarray(circuito_estado(parametros), dtype=complex)
    energy_exact, state_exact = _estado_exacto_embebido(
        n_sites, hop, U, n_up, n_down
    )
    energy_vqe = float(circuito_fisico(parametros))
    fidelity = float(abs(np.vdot(state_exact, state_vqe)) ** 2)
    observables_vqe = observables_estado_pennylane(state_vqe, n_sites)
    observables_exactos = observables_estado_pennylane(
        state_exact, n_sites
    )
    return {
        "U/J": U / hop,
        "energia_vqe": energy_vqe,
        "energia_exacta": energy_exact,
        "error_energia": abs(energy_vqe - energy_exact),
        "fidelidad": fidelity,
        "peso_sector": float(circuito_sector(parametros)),
        "parametros_optimos": np.asarray(parametros),
        "historial": np.asarray(history),
        "estado_vqe": state_vqe,
        "estado_exacto": state_exact,
        "observables_vqe": observables_vqe,
        "observables_exactos": observables_exactos,
        "pasos": len(history) - 1,
    }


def barrido_vqe_FH(
    valores_U_sobre_J: Sequence[float],
    hop: float = 1.0,
    n_sites: int = 2,
    n_layers: int = 3,
    learning_rate: float = 0.06,
    max_steps: int = 350,
    seed: int = 7,
):
    """Compara VQE y diagonalización exacta para un barrido de U/J."""
    rows = []
    details = {}
    parameters = None
    for index, ratio in enumerate(valores_U_sobre_J):
        result = vqe_FH_pennylane(
            U=float(ratio) * hop,
            hop=hop,
            n_sites=n_sites,
            n_layers=n_layers,
            learning_rate=learning_rate,
            max_steps=max_steps,
            seed=seed + index,
            parametros_iniciales=parameters,
        )
        parameters = result["parametros_optimos"]
        rows.append(
            {
                "U/J": result["U/J"],
                "E_VQE": result["energia_vqe"],
                "E_exacta": result["energia_exacta"],
                "error_E": result["error_energia"],
                "fidelidad": result["fidelidad"],
                "peso_sector": result["peso_sector"],
                "D_VQE": result["observables_vqe"]["doble_ocupacion"],
                "D_exacta": result["observables_exactos"]["doble_ocupacion"],
                "S_pi_VQE": result["observables_vqe"][
                    "estructura_spin_pi"
                ],
                "S_pi_exacta": result["observables_exactos"][
                    "estructura_spin_pi"
                ],
                "error_densidad": float(
                    np.max(
                        np.abs(
                            result["observables_vqe"]["densidad"]
                            - result["observables_exactos"]["densidad"]
                        )
                    )
                ),
                "pasos": result["pasos"],
            }
        )
        details[float(ratio)] = result
    return rows, details

