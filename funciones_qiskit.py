"""Implementaciones cuánticas reutilizables del proyecto Quantathon.

Este módulo reúne las funciones Qiskit que antes vivían dentro de varios
notebooks: TFIM, evolución de Trotter y la extensión 1D de Fermi-Hubbard.
Las funciones de presentación se mantienen en ``Funciones.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def _enlaces_tfim(n_q: int, periodic: bool) -> list[tuple[int, int]]:
    if n_q < 2:
        raise ValueError("n_q debe ser al menos 2.")
    enlaces = [(i, i + 1) for i in range(n_q - 1)]
    if periodic and n_q > 2:
        enlaces.append((n_q - 1, 0))
    return enlaces


def hamiltoniano_tfim(
    J: float,
    h: float,
    n_q: int,
    periodic: bool = True,
) -> SparsePauliOp:
    """Construye H = -J sum(ZZ) - h sum(X)."""
    terminos = [
        ("ZZ", [i, j], -J)
        for i, j in _enlaces_tfim(n_q, periodic)
    ]
    terminos.extend(("X", [i], -h) for i in range(n_q))
    return SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)


def operadores_observables_tfim(
    n_q: int,
    periodic: bool = True,
) -> dict[str, SparsePauliOp]:
    """Devuelve Mz, Mz², Mx y la correlación media Czz."""
    enlaces = _enlaces_tfim(n_q, periodic)
    mz = SparsePauliOp.from_sparse_list(
        [("Z", [i], 1.0 / n_q) for i in range(n_q)],
        num_qubits=n_q,
    )
    mx = SparsePauliOp.from_sparse_list(
        [("X", [i], 1.0 / n_q) for i in range(n_q)],
        num_qubits=n_q,
    )
    czz = SparsePauliOp.from_sparse_list(
        [("ZZ", [i, j], 1.0 / len(enlaces)) for i, j in enlaces],
        num_qubits=n_q,
    )
    return {"mz": mz, "mz2": mz.compose(mz), "mx": mx, "czz": czz}


def medir_observables_tfim(
    estado: Statevector | np.ndarray,
    n_q: int,
    periodic: bool = True,
) -> dict[str, float]:
    """Calcula los observables TFIM sobre un vector de estado."""
    vector = estado.data if isinstance(estado, Statevector) else estado
    psi = Statevector(np.ascontiguousarray(vector, dtype=complex))
    return {
        nombre: float(np.real_if_close(psi.expectation_value(operador)))
        for nombre, operador in operadores_observables_tfim(n_q, periodic).items()
    }


def estado_base_tfim(
    J: float,
    h: float,
    n_q: int,
    periodic: bool = True,
) -> tuple[float, Statevector]:
    """Obtiene energía y estado base mediante diagonalización exacta."""
    matriz = np.asarray(
        hamiltoniano_tfim(J, h, n_q, periodic).to_matrix(),
        dtype=complex,
    )
    energias, estados = np.linalg.eigh(matriz)
    return (
        float(np.real(energias[0])),
        Statevector(np.ascontiguousarray(estados[:, 0], dtype=complex)),
    )


def evolucion_exacta_tfim(
    t: float,
    h: float,
    J: float,
    n_q: int,
    periodic: bool = True,
    estado_inicial: Statevector | None = None,
) -> Statevector:
    """Evoluciona exactamente un estado al diagonalizar el Hamiltoniano."""
    if t < 0:
        raise ValueError("t debe ser mayor o igual que cero.")
    matriz = np.asarray(
        hamiltoniano_tfim(J, h, n_q, periodic).to_matrix(),
        dtype=complex,
    )
    energias, autovectores = np.linalg.eigh(matriz)
    psi_0 = (
        Statevector.from_label("0" * n_q)
        if estado_inicial is None
        else estado_inicial
    )
    coeficientes = autovectores.conj().T @ np.asarray(psi_0.data)
    psi_t = autovectores @ (np.exp(-1j * energias * t) * coeficientes)
    return Statevector(np.ascontiguousarray(psi_t, dtype=complex))


def circuito_trotter_tfim(
    J: float,
    h: float,
    n_q: int,
    r: int,
    t: float,
    periodic: bool = True,
) -> QuantumCircuit:
    """Circuito Suzuki-Trotter de primer orden para el TFIM."""
    if r < 1:
        raise ValueError("r debe ser al menos 1.")
    if t < 0:
        raise ValueError("t debe ser mayor o igual que cero.")
    dt = t / r
    circuito = QuantumCircuit(n_q)
    enlaces = _enlaces_tfim(n_q, periodic)
    for _ in range(r):
        for i, j in enlaces:
            circuito.rzz(-2.0 * J * dt, i, j)
        for i in range(n_q):
            circuito.rx(-2.0 * h * dt, i)
    return circuito


def evolucion_trotter_tfim(
    J: float,
    h: float,
    n_q: int,
    r: int,
    t: float,
    periodic: bool = True,
) -> Statevector:
    """Devuelve el estado de ``circuito_trotter_tfim``."""
    circuito = circuito_trotter_tfim(J, h, n_q, r, t, periodic)
    return Statevector.from_instruction(circuito)


def fidelidad_estados(
    estado_a: Statevector | np.ndarray,
    estado_b: Statevector | np.ndarray,
) -> float:
    """F = |<a|b>|²."""
    a = np.asarray(estado_a.data if isinstance(estado_a, Statevector) else estado_a)
    b = np.asarray(estado_b.data if isinstance(estado_b, Statevector) else estado_b)
    return float(np.abs(np.vdot(a, b)) ** 2)


def barrido_estado_base_tfim(
    n_q: int,
    valores_h_sobre_J: Sequence[float],
    J: float = 1.0,
    periodic: bool = True,
) -> dict[str, np.ndarray]:
    """Línea base ED del estado fundamental."""
    x = np.asarray(valores_h_sobre_J, dtype=float)
    datos = {
        "h_sobre_J": x,
        "energia_por_sitio": np.empty(len(x)),
        "mz": np.empty(len(x)),
        "mz2": np.empty(len(x)),
        "mx": np.empty(len(x)),
        "czz": np.empty(len(x)),
    }
    for k, razon in enumerate(x):
        energia, estado = estado_base_tfim(J, razon * J, n_q, periodic)
        observables = medir_observables_tfim(estado, n_q, periodic)
        datos["energia_por_sitio"][k] = energia / n_q
        for nombre in ("mz", "mz2", "mx", "czz"):
            datos[nombre][k] = observables[nombre]
    return datos


def dinamica_tfim(
    J: float,
    h: float,
    n_q: int,
    tiempos: Sequence[float],
    r: int,
    periodic: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Compara ED y Trotter desde |00...0> en una malla temporal."""
    tiempos = np.asarray(tiempos, dtype=float)
    salida: dict[str, np.ndarray | float | int] = {
        "tiempos": tiempos,
        "J": J,
        "h": h,
        "n_q": n_q,
        "r": r,
    }
    for metodo in ("exacta", "trotter"):
        for observable in ("mz", "mx", "czz"):
            salida[f"{observable}_{metodo}"] = np.empty(len(tiempos))
    salida["fidelidad"] = np.empty(len(tiempos))

    for k, tiempo in enumerate(tiempos):
        exacta = evolucion_exacta_tfim(
            float(tiempo), h, J, n_q, periodic
        )
        trotter = evolucion_trotter_tfim(
            J, h, n_q, r, float(tiempo), periodic
        )
        obs_exactos = medir_observables_tfim(exacta, n_q, periodic)
        obs_trotter = medir_observables_tfim(trotter, n_q, periodic)
        for observable in ("mz", "mx", "czz"):
            salida[f"{observable}_exacta"][k] = obs_exactos[observable]
            salida[f"{observable}_trotter"][k] = obs_trotter[observable]
        salida["fidelidad"][k] = fidelidad_estados(exacta, trotter)
    return salida


def analisis_convergencia_trotter(
    n_q: int,
    valores_r: Iterable[int],
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = True,
) -> dict[str, np.ndarray]:
    """Evalúa fidelidad y error de observables al reducir Δt."""
    pasos = np.asarray(list(valores_r), dtype=int)
    exacta = evolucion_exacta_tfim(t, h, J, n_q, periodic)
    obs_exactos = medir_observables_tfim(exacta, n_q, periodic)
    fidelidades = np.empty(len(pasos))
    error_max_observable = np.empty(len(pasos))
    for k, r in enumerate(pasos):
        aproximada = evolucion_trotter_tfim(J, h, n_q, int(r), t, periodic)
        obs_trotter = medir_observables_tfim(aproximada, n_q, periodic)
        fidelidades[k] = fidelidad_estados(exacta, aproximada)
        error_max_observable[k] = max(
            abs(obs_trotter[nombre] - obs_exactos[nombre])
            for nombre in ("mz", "mx", "czz")
        )
    return {
        "r": pasos,
        "dt": t / pasos,
        "fidelidad": fidelidades,
        "infidelidad": 1.0 - fidelidades,
        "error_max_observable": error_max_observable,
    }


def analisis_escalado_tfim(
    tamanos: Sequence[int],
    r: int = 10,
    t: float = 1.0,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = True,
) -> dict[str, np.ndarray]:
    """Compara fidelidad y recursos al crecer el número de espines."""
    tamanos = np.asarray(tamanos, dtype=int)
    fidelidades = np.empty(len(tamanos))
    profundidades = np.empty(len(tamanos), dtype=int)
    compuertas = np.empty(len(tamanos), dtype=int)
    for k, n_q in enumerate(tamanos):
        exacta = evolucion_exacta_tfim(t, h, J, int(n_q), periodic)
        circuito = circuito_trotter_tfim(J, h, int(n_q), r, t, periodic)
        aproximada = Statevector.from_instruction(circuito)
        fidelidades[k] = fidelidad_estados(exacta, aproximada)
        profundidades[k] = circuito.depth()
        compuertas[k] = circuito.size()
    return {
        "n_q": tamanos,
        "fidelidad": fidelidades,
        "profundidad": profundidades,
        "compuertas": compuertas,
    }


# ---------------------------------------------------------------------------
# Funciones rescatadas: ejemplos introductorios de Trotter
# ---------------------------------------------------------------------------

def evolucion_un_qubit_z(t: float, omega: float = 1.0) -> Statevector:
    circuito = QuantumCircuit(1)
    circuito.h(0)
    circuito.rz(omega * t, 0)
    return Statevector.from_instruction(circuito)


def trotter_un_qubit(
    t: float,
    r: int,
    a: float = 1.0,
    b: float = 1.0,
    orden: int = 1,
) -> Statevector:
    """Evolución de H=aX+bZ, de primer o segundo orden."""
    if r < 1:
        raise ValueError("r debe ser al menos 1.")
    if orden not in (1, 2):
        raise ValueError("orden debe ser 1 o 2.")
    dt = t / r
    circuito = QuantumCircuit(1)
    circuito.h(0)
    for _ in range(r):
        if orden == 1:
            circuito.rx(2 * a * dt, 0)
            circuito.rz(2 * b * dt, 0)
        else:
            circuito.rx(a * dt, 0)
            circuito.rz(2 * b * dt, 0)
            circuito.rx(a * dt, 0)
    return Statevector.from_instruction(circuito)


# ---------------------------------------------------------------------------
# Funciones rescatadas: Fermi-Hubbard 1D con Jordan-Wigner
# ---------------------------------------------------------------------------

def r_xzx(
    circuito: QuantumCircuit,
    theta: float,
    q0: int,
    q1: int,
    q2: int,
) -> None:
    """Implementa exp(-i theta XZX / 2)."""
    circuito.h(q0)
    circuito.h(q2)
    circuito.cx(q0, q2)
    circuito.cx(q1, q2)
    circuito.rz(theta, q2)
    circuito.cx(q1, q2)
    circuito.cx(q0, q2)
    circuito.h(q2)
    circuito.h(q0)


def r_yzy(
    circuito: QuantumCircuit,
    theta: float,
    q0: int,
    q1: int,
    q2: int,
) -> None:
    """Implementa exp(-i theta YZY / 2)."""
    for q in (q0, q2):
        circuito.sdg(q)
        circuito.h(q)
    circuito.cx(q0, q2)
    circuito.cx(q1, q2)
    circuito.rz(theta, q2)
    circuito.cx(q1, q2)
    circuito.cx(q0, q2)
    for q in (q2, q0):
        circuito.h(q)
        circuito.s(q)


def aplicar_hopping_enlace(
    circuito: QuantumCircuit,
    hop: float,
    dt: float,
    i: int,
) -> None:
    """Aplica el hopping entre los sitios físicos i e i+1."""
    theta = -hop * dt
    q_up_i, q_down_i = 2 * i, 2 * i + 1
    q_up_j, q_down_j = 2 * i + 2, 2 * i + 3
    r_xzx(circuito, theta, q_up_i, q_down_i, q_up_j)
    r_yzy(circuito, theta, q_up_i, q_down_i, q_up_j)
    r_xzx(circuito, theta, q_down_i, q_up_j, q_down_j)
    r_yzy(circuito, theta, q_down_i, q_up_j, q_down_j)


def aplicar_interaccion_sitio(
    circuito: QuantumCircuit,
    U: float,
    dt: float,
    i: int,
) -> None:
    """Aplica U n_up n_down omitiendo una fase global."""
    q_up, q_down = 2 * i, 2 * i + 1
    circuito.rz(-U * dt / 2, q_up)
    circuito.rz(-U * dt / 2, q_down)
    circuito.rzz(U * dt / 2, q_up, q_down)


def circuito_fermi_hubbard_1d(
    hop: float,
    U: float,
    n_q: int,
    r: int,
    t: float,
    occupied_modes: Sequence[int] | None = None,
) -> QuantumCircuit:
    """Circuito Trotter de primer orden para una cadena abierta 1D."""
    if n_q < 4 or n_q % 2:
        raise ValueError("n_q debe ser par y al menos 4.")
    if r < 1:
        raise ValueError("r debe ser al menos 1.")
    n_sites = n_q // 2
    dt = t / r
    circuito = QuantumCircuit(n_q)
    for q in occupied_modes or ():
        if q < 0 or q >= n_q:
            raise ValueError(f"El modo {q} no existe.")
        circuito.x(q)
    for _ in range(r):
        for i in range(0, n_sites - 1, 2):
            aplicar_hopping_enlace(circuito, hop, dt, i)
        for i in range(1, n_sites - 1, 2):
            aplicar_hopping_enlace(circuito, hop, dt, i)
        for i in range(n_sites):
            aplicar_interaccion_sitio(circuito, U, dt, i)
    return circuito


def evolucion_fermi_hubbard_1d(
    hop: float,
    U: float,
    n_q: int,
    r: int,
    t: float,
    occupied_modes: Sequence[int] | None = None,
) -> Statevector:
    return Statevector.from_instruction(
        circuito_fermi_hubbard_1d(hop, U, n_q, r, t, occupied_modes)
    )


def ocupaciones_modos(estado: Statevector) -> np.ndarray:
    """Calcula <n_q> para cada modo fermiónico."""
    probabilidades = estado.probabilities()
    ocupaciones = np.zeros(estado.num_qubits)
    for indice, probabilidad in enumerate(probabilidades):
        for q in range(estado.num_qubits):
            ocupaciones[q] += ((indice >> q) & 1) * probabilidad
    return ocupaciones


def densidad_por_sitio(estado: Statevector) -> np.ndarray:
    """Devuelve <n_up+n_down> por sitio físico."""
    ocupaciones = ocupaciones_modos(estado)
    return ocupaciones.reshape(-1, 2).sum(axis=1)


def doble_ocupacion_por_sitio(estado: Statevector) -> np.ndarray:
    """Calcula <n_up n_down> por sitio."""
    probabilidades = estado.probabilities()
    dobles = np.zeros(estado.num_qubits // 2)
    for indice, probabilidad in enumerate(probabilidades):
        for sitio in range(len(dobles)):
            n_up = (indice >> (2 * sitio)) & 1
            n_down = (indice >> (2 * sitio + 1)) & 1
            dobles[sitio] += n_up * n_down * probabilidad
    return dobles


def doble_ocupacion_promedio(estado: Statevector) -> float:
    return float(np.mean(doble_ocupacion_por_sitio(estado)))


def evolucion_fermi_hubbard_observables(
    hop: float,
    U: float,
    n_q: int,
    r: int,
    tiempos: Sequence[float],
    occupied_modes: Sequence[int],
) -> dict[str, np.ndarray]:
    """Densidad, doble ocupación y número de partículas vs. tiempo."""
    tiempos = np.asarray(tiempos, dtype=float)
    densidades = np.empty((len(tiempos), n_q // 2))
    dobles = np.empty(len(tiempos))
    numero_particulas = np.empty(len(tiempos))
    for k, tiempo in enumerate(tiempos):
        estado = evolucion_fermi_hubbard_1d(
            hop, U, n_q, r, float(tiempo), occupied_modes
        )
        densidades[k] = densidad_por_sitio(estado)
        dobles[k] = doble_ocupacion_promedio(estado)
        numero_particulas[k] = np.sum(densidades[k])
    return {
        "tiempos": tiempos,
        "densidades": densidades,
        "doble_ocupacion": dobles,
        "numero_particulas": numero_particulas,
    }


# Alias compatibles con los notebooks originales.
H_FH_trott1_o = evolucion_fermi_hubbard_1d
evolucion_FH = evolucion_fermi_hubbard_observables
evolve_tfim_exact = evolucion_exacta_tfim
circuito_tfim_trotter_cerrado = circuito_trotter_tfim
evolucion_tfim_trotter_cerrada = evolucion_trotter_tfim
