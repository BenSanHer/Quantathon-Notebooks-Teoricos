"""Funciones para la simulación Trotter de Fermi-Hubbard en una cadena 1D.

Estas funciones fueron extraídas de ``Trotterizacion Fermi Hubbard.ipynb``
para que el notebook contenga únicamente la explicación y los experimentos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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

