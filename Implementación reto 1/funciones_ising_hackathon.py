"""
Funciones extraídas de los cuatro notebooks del proyecto TFIM/Ising.

Las funciones con nombres repetidos fueron renombradas para que ninguna
definición se pierda al importarlas desde este módulo.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector



# ======================================================================================
# FUNCIONES EXTRAÍDAS DE: 1 Trotterización(2).ipynb
# ======================================================================================

# Celda 10 · nombre original: evolve_with_z
def evolve_with_z(t: float, omega: float=1.0) -> Statevector:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(omega * t, 0)
    return Statevector.from_instruction(circuit)

# Celda 14 · nombre original: trotterized_hamiltonian_primer_orden
def trotterized_hamiltonian_primer_orden(circ: QuantumCircuit, dt: float, r: int, a: complex, b: complex) -> None:
    for _ in range(r):
        circ.rx(2 * a * dt, 0)
        circ.rz(2 * b * dt, 0)

# Celda 14 · nombre original: evolve_trotter_primer_orden
def evolve_trotter_primer_orden(t: float, r: int, omega: float=1.0) -> Statevector:
    dt = t / r
    circuit = QuantumCircuit(1)
    circuit.h(0)
    trotterized_hamiltonian_primer_orden(circuit, dt, r, 1.0, 1.0)
    return Statevector.from_instruction(circuit)

# Celda 16 · nombre original: trotterized_hamiltonian_segundo_orden
def trotterized_hamiltonian_segundo_orden(circ: QuantumCircuit, dt: float, r: int, a: complex, b: complex) -> None:
    for _ in range(r):
        circ.rx(a * dt, 0)
        circ.rz(2 * b * dt, 0)
        circ.rx(a * dt, 0)

# Celda 16 · nombre original: evolve_trotter_segundo_orden
def evolve_trotter_segundo_orden(t: float, r: int, omega: float=1.0) -> Statevector:
    dt = t / r
    circuit = QuantumCircuit(1)
    circuit.h(0)
    trotterized_hamiltonian_primer_orden(circuit, dt, r, 1.0, 1.0)
    return Statevector.from_instruction(circuit)



# ======================================================================================
# FUNCIONES EXTRAÍDAS DE: 2 Modelo de Ising con campo transverso(1).ipynb
# ======================================================================================

# Celda 8 · nombre original: evolve_tfim_exact
def evolve_tfim_exact(t: float, h: float, J: float, n_qubits: int, r: int, periodic: bool=False) -> Statevector:
    if n_qubits < 1:
        raise ValueError('n_qubits debe ser al menos 1.')
    if r < 1:
        raise ValueError('r debe ser al menos 1.')
    if t < 0:
        raise ValueError('t debe ser mayor o igual que cero.')
    if periodic and n_qubits < 2:
        raise ValueError('Las condiciones periódicas requieren al menos 2 qubits.')
    pauli_terms = []
    for qubit in range(n_qubits):
        label = ['I'] * n_qubits
        label[n_qubits - 1 - qubit] = 'X'
        pauli_string = ''.join(label)
        pauli_terms.append((pauli_string, -h))
    if periodic and n_qubits > 2:
        pairs = [(qubit, (qubit + 1) % n_qubits) for qubit in range(n_qubits)]
    else:
        pairs = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    for qubit_1, qubit_2 in pairs:
        label = ['I'] * n_qubits
        label[n_qubits - 1 - qubit_1] = 'Z'
        label[n_qubits - 1 - qubit_2] = 'Z'
        pauli_string = ''.join(label)
        pauli_terms.append((pauli_string, -J))
    hamiltonian_operator = SparsePauliOp.from_list(pauli_terms)
    hamiltonian_matrix = hamiltonian_operator.to_matrix()
    eigenvalues, eigenvectors = eigh(hamiltonian_matrix)
    dimension = 2 ** n_qubits
    initial_state = np.zeros(dimension, dtype=complex)
    initial_state[0] = 1.0
    coefficients = eigenvectors.conj().T @ initial_state
    evolved_coefficients = np.exp(-1j * eigenvalues * t) * coefficients
    evolved_state = eigenvectors @ evolved_coefficients
    return Statevector(evolved_state)

# Celda 12 · nombre original: H_trott1_c
def H_trott1_c(J: float, h: float, n_q: int, r: int, t: float) -> Statevector:
    dt = t / r
    qc = QuantumCircuit(n_q)
    for _ in range(r):
        for i in range(n_q):
            qc.rzz(-2 * J * dt, i - 1, i)
            qc.rx(-2 * h * dt, i - 1)
    return Statevector.from_instruction(qc)

# Celda 12 · nombre original: H_trott1_o
def H_trott1_o(J: float, h: float, n_q: int, r: int, t: float) -> Statevector:
    dt = t / r
    qc = QuantumCircuit(n_q)
    for _ in range(r):
        for i in range(1, n_q):
            qc.rzz(-2 * J * dt, i - 1, i)
            qc.rx(-2 * h * dt, i - 1)
        qc.rx(-2 * h * dt, n_q - 1)
    return Statevector.from_instruction(qc)



# ======================================================================================
# FUNCIONES EXTRAÍDAS DE: 3 Gráficas bajo diagonalización directa.ipynb
# ======================================================================================

# Celda 0 · nombre original: estado_base_ising_con_energia
def estado_base_ising_con_energia(J: float, h: float, n_q: int, periodic: bool=False) -> tuple[float, Statevector]:
    if n_q < 1:
        raise ValueError('n_q debe ser al menos 1.')
    terminos = []
    for i in range(n_q - 1):
        terminos.append(('ZZ', [i, i + 1], -J))
    if periodic and n_q > 2:
        terminos.append(('ZZ', [n_q - 1, 0], -J))
    for i in range(n_q):
        terminos.append(('X', [i], -h))
    H = SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)
    eigenvalues, eigenvectors = np.linalg.eigh(H.to_matrix())
    energia_base = float(eigenvalues[0])
    estado_base = Statevector(eigenvectors[:, 0])
    return (energia_base, estado_base)

# Celda 2 · nombre original: hamiltoniano_ising
def hamiltoniano_ising(J: float, h: float, n_q: int, periodic: bool=False) -> SparsePauliOp:
    """
    Hamiltoniano de Ising con campo transverso:

        H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i
    """
    if n_q < 2:
        raise ValueError('n_q debe ser al menos 2.')
    terminos = []
    for i in range(n_q - 1):
        terminos.append(('ZZ', [i, i + 1], -J))
    if periodic and n_q > 2:
        terminos.append(('ZZ', [n_q - 1, 0], -J))
    for i in range(n_q):
        terminos.append(('X', [i], -h))
    return SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)

# Celda 2 · nombre original: estado_base_ising
def estado_base_ising(J: float, h: float, n_q: int, periodic: bool=False) -> Statevector:
    """
    Obtiene el estado base mediante diagonalización exacta.
    """
    H = hamiltoniano_ising(J=J, h=h, n_q=n_q, periodic=periodic)
    H_matrix = np.asarray(H.to_matrix(), dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state = np.ascontiguousarray(eigenvectors[:, 0], dtype=complex)
    return Statevector(ground_state)

# Celda 2 · nombre original: magnetizacion_z_cuadrada
def magnetizacion_z_cuadrada(n_q: int) -> SparsePauliOp:
    """
    Construye el operador:

                  1
        M_z = -------- Σ_i Z_i
                 n_q

    y devuelve M_z².

    Usamos:

        M_z² = I/n_q + (2/n_q²) Σ_{i<j} Z_i Z_j
    """
    terminos = [('I' * n_q, 1.0 / n_q)]
    for i in range(n_q):
        for j in range(i + 1, n_q):
            pauli = ['I'] * n_q
            pauli[n_q - 1 - i] = 'Z'
            pauli[n_q - 1 - j] = 'Z'
            terminos.append((''.join(pauli), 2.0 / n_q ** 2))
    return SparsePauliOp.from_list(terminos)

# Celda 2 · nombre original: barrido_magnetizacion
def barrido_magnetizacion(n_q: int, J: float=1.0, h_min: float=0.0, h_max: float=2.0, puntos: int=101, periodic: bool=False):
    """
    Calcula <M_z²> para diferentes valores de h/J.
    """
    if J == 0:
        raise ValueError('J no puede ser cero porque graficamos h/J.')
    valores_h = np.linspace(h_min, h_max, puntos)
    valores_h_sobre_J = valores_h / J
    Mz2 = magnetizacion_z_cuadrada(n_q)
    valores_mz2 = []
    for h in valores_h:
        psi_0 = estado_base_ising(J=J, h=h, n_q=n_q, periodic=periodic)
        esperanza = psi_0.expectation_value(Mz2)
        esperanza = np.real_if_close(esperanza)
        valores_mz2.append(float(np.real(esperanza)))
    return (valores_h_sobre_J, np.array(valores_mz2))

# Celda 2 · nombre original: graficar_magnetizacion
def graficar_magnetizacion(n_q: int, puntos: int=101, periodic: bool=False):
    J = 1.0
    h_sobre_J, mz2 = barrido_magnetizacion(n_q=n_q, J=J, h_min=0.0, h_max=2.0, puntos=puntos, periodic=periodic)
    plt.figure(figsize=(8, 5))
    plt.plot(h_sobre_J, mz2, marker='o', markersize=3, linewidth=1.5, label=f'$N={n_q}$')
    plt.axvline(x=1.0, linestyle='--', label='Punto crítico $h/J=1$')
    plt.xlabel('$h/J$')
    plt.ylabel('$\\langle M_z^2\\rangle$')
    plt.title('Magnetización longitudinal cuadrática\nModelo de Ising con campo transverso')
    plt.xlim(0, 2)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Celda 4 · nombre original: magnetizacion_x
def magnetizacion_x(n_q: int) -> SparsePauliOp:
    """
    Operador de magnetización transversal promedio:
        M_x = (1 / n_q) Σ_i X_i
    """
    terminos = [('X', [i], 1.0 / n_q) for i in range(n_q)]
    return SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)

# Celda 4 · nombre original: correlacion_zz_vecinos
def correlacion_zz_vecinos(n_q: int, periodic: bool=False) -> SparsePauliOp:
    """
    Correlación promedio entre primeros vecinos:

        C_zz = (1 / N_b) Σ_<i,j> Z_i Z_j

    donde N_b es el número de enlaces.
    """
    if n_q < 2:
        raise ValueError('n_q debe ser al menos 2.')
    terminos = [('ZZ', [i, i + 1], 1.0) for i in range(n_q - 1)]
    if periodic:
        terminos.append(('ZZ', [n_q - 1, 0], 1.0))
    numero_enlaces = len(terminos)
    terminos_normalizados = [(pauli, indices, coeficiente / numero_enlaces) for pauli, indices, coeficiente in terminos]
    return SparsePauliOp.from_sparse_list(terminos_normalizados, num_qubits=n_q)

# Celda 4 · nombre original: barrido_magnetizacion_x_y_correlacion_zz
def barrido_magnetizacion_x_y_correlacion_zz(n_q: int, J: float=1.0, h_min: float=0.0, h_max: float=2.0, puntos: int=101, periodic: bool=False):
    if J == 0:
        raise ValueError('J no puede ser cero porque graficamos h/J.')
    valores_h = np.linspace(h_min, h_max, puntos)
    valores_h_sobre_J = valores_h / J
    Mx = magnetizacion_x(n_q)
    Czz = correlacion_zz_vecinos(n_q=n_q, periodic=periodic)
    valores_mx = []
    valores_czz = []
    for h in valores_h:
        psi_0 = estado_base_ising(J=J, h=h, n_q=n_q, periodic=periodic)
        mx = psi_0.expectation_value(Mx)
        czz = psi_0.expectation_value(Czz)
        valores_mx.append(float(np.real(np.real_if_close(mx))))
        valores_czz.append(float(np.real(np.real_if_close(czz))))
    return (valores_h_sobre_J, np.asarray(valores_mx), np.asarray(valores_czz))

# Celda 4 · nombre original: graficar_magnetizacion_x
def graficar_magnetizacion_x(n_q: int, puntos: int=101, periodic: bool=False):
    h_sobre_J, mx, _ = barrido_magnetizacion_x_y_correlacion_zz(n_q=n_q, J=1.0, h_min=0.0, h_max=2.0, puntos=puntos, periodic=periodic)
    plt.figure(figsize=(8, 5))
    plt.plot(h_sobre_J, mx, marker='o', markersize=3, linewidth=1.5, label=f'$N={n_q}$')
    plt.axvline(1.0, linestyle='--', label='Punto crítico $h/J=1$')
    plt.xlabel('$h/J$')
    plt.ylabel('$\\langle M_x\\rangle$')
    plt.title('Magnetización transversal\nModelo de Ising con campo transverso')
    plt.xlim(0, 2)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Celda 4 · nombre original: graficar_correlacion_zz
def graficar_correlacion_zz(n_q: int, puntos: int=101, periodic: bool=False):
    h_sobre_J, _, czz = barrido_magnetizacion_x_y_correlacion_zz(n_q=n_q, J=1.0, h_min=0.0, h_max=2.0, puntos=puntos, periodic=periodic)
    plt.figure(figsize=(8, 5))
    plt.plot(h_sobre_J, czz, marker='o', markersize=3, linewidth=1.5, label=f'$N={n_q}$')
    plt.axvline(1.0, linestyle='--', label='Punto crítico $h/J=1$')
    plt.xlabel('$h/J$')
    plt.ylabel('$\\frac{1}{N_b}\\sum_{\\langle i,j\\rangle}\\langle Z_iZ_j\\rangle$')
    plt.title('Correlación longitudinal entre vecinos\nModelo de Ising con campo transverso')
    plt.xlim(0, 2)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ======================================================================================
# FUNCIONES EXTRAÍDAS DE: 3_Graficas_Diagonalizacio_directa_ordenado.ipynb
# ======================================================================================

# Celda 2 · nombre original: hamiltoniano_ising_cerrado
def hamiltoniano_ising_cerrado(J: float, h: float, n_q: int) -> SparsePauliOp:
    """
    Hamiltoniano de Ising con campo transverso y frontera periódica:

        H = -J Σ_i Z_i Z_{i+1} - h Σ_i X_i,

    con Z_N = Z_0.
    """
    if n_q < 2:
        raise ValueError('n_q debe ser al menos 2.')
    terminos = []
    for i in range(n_q):
        j = (i + 1) % n_q
        terminos.append(('ZZ', [i, j], -J))
    for i in range(n_q):
        terminos.append(('X', [i], -h))
    return SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)

# Celda 2 · nombre original: estado_base_ising_cerrado
def estado_base_ising_cerrado(J: float, h: float, n_q: int, devolver_energia: bool=False):
    """Obtiene el estado base por diagonalización exacta."""
    H = hamiltoniano_ising_cerrado(J=J, h=h, n_q=n_q)
    H_matrix = np.asarray(H.to_matrix(), dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    vector_base = np.ascontiguousarray(eigenvectors[:, 0], dtype=complex)
    psi_0 = Statevector(vector_base)
    if devolver_energia:
        return (float(np.real(eigenvalues[0])), psi_0)
    return psi_0

# Celda 3 · nombre original: magnetizacion_z_cuadrada_cerrada
def magnetizacion_z_cuadrada_cerrada(n_q: int) -> SparsePauliOp:
    """
    M_z², donde M_z = (1/N) Σ_i Z_i.

    M_z² = I/N + (2/N²) Σ_{i<j} Z_i Z_j.
    """
    terminos = [('I' * n_q, 1.0 / n_q)]
    for i in range(n_q):
        for j in range(i + 1, n_q):
            pauli = ['I'] * n_q
            pauli[n_q - 1 - i] = 'Z'
            pauli[n_q - 1 - j] = 'Z'
            terminos.append((''.join(pauli), 2.0 / n_q ** 2))
    return SparsePauliOp.from_list(terminos)

# Celda 3 · nombre original: magnetizacion_x_cerrada
def magnetizacion_x_cerrada(n_q: int) -> SparsePauliOp:
    """M_x = (1/N) Σ_i X_i."""
    return SparsePauliOp.from_sparse_list([('X', [i], 1.0 / n_q) for i in range(n_q)], num_qubits=n_q)

# Celda 3 · nombre original: correlacion_zz_vecinos_cerrada
def correlacion_zz_vecinos_cerrada(n_q: int) -> SparsePauliOp:
    """
    C_zz = (1/N) Σ_i Z_i Z_{i+1}, con Z_N = Z_0.
    """
    terminos = [('ZZ', [i, (i + 1) % n_q], 1.0 / n_q) for i in range(n_q)]
    return SparsePauliOp.from_sparse_list(terminos, num_qubits=n_q)

# Celda 4 · nombre original: barrido_observables_ising_cerrado
def barrido_observables_ising_cerrado(n_q: int, J: float=1.0, h_min: float=0.0, h_max: float=2.0, puntos: int=101):
    """
    Calcula los tres observables en un único barrido.

    Devuelve:
        h/J, <Mz²>, <Mx>, <Czz>, E0/N
    """
    if J == 0:
        raise ValueError('J no puede ser cero porque se grafica h/J.')
    if puntos < 2:
        raise ValueError('puntos debe ser al menos 2.')
    valores_h = np.linspace(h_min, h_max, puntos)
    Mz2 = magnetizacion_z_cuadrada_cerrada(n_q)
    Mx = magnetizacion_x_cerrada(n_q)
    Czz = correlacion_zz_vecinos_cerrada(n_q)
    valores_mz2 = np.empty(puntos)
    valores_mx = np.empty(puntos)
    valores_czz = np.empty(puntos)
    energias_por_sitio = np.empty(puntos)
    for k, h in enumerate(valores_h):
        energia_0, psi_0 = estado_base_ising_cerrado(J=J, h=h, n_q=n_q, devolver_energia=True)
        valores_mz2[k] = float(np.real_if_close(psi_0.expectation_value(Mz2)))
        valores_mx[k] = float(np.real_if_close(psi_0.expectation_value(Mx)))
        valores_czz[k] = float(np.real_if_close(psi_0.expectation_value(Czz)))
        energias_por_sitio[k] = energia_0 / n_q
    return {'h_sobre_J': valores_h / J, 'mz2': valores_mz2, 'mx': valores_mx, 'czz': valores_czz, 'energia_por_sitio': energias_por_sitio}

# Celda 5 · nombre original: graficar_observables_ising
def graficar_observables_ising(resultados: dict, n_q: int, mostrar_energia: bool=False):
    """Presenta los observables en paneles comparables."""
    x = resultados['h_sobre_J']
    observables = [(resultados['mz2'], '$\\langle M_z^2\\rangle$', 'Orden longitudinal'), (resultados['mx'], '$\\langle M_x\\rangle$', 'Magnetización transversal'), (resultados['czz'], '$\\langle C_{zz}\\rangle$', 'Correlación entre vecinos')]
    if mostrar_energia:
        observables.append((resultados['energia_por_sitio'], '$E_0/N$', 'Energía base por sitio'))
    n_paneles = len(observables)
    fig, axes = plt.subplots(1, n_paneles, figsize=(5.2 * n_paneles, 4.5), sharex=True)
    if n_paneles == 1:
        axes = [axes]
    markevery = max(1, len(x) // 20)
    for ax, (y, ylabel, titulo) in zip(axes, observables):
        ax.plot(x, y, linewidth=2, marker='o', markersize=4, markevery=markevery, label=f'$N={n_q}$')
        ax.axvline(1.0, linestyle='--', linewidth=1.5, label='$h/J=1$')
        ax.set_title(titulo)
        ax.set_xlabel('$h/J$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(x.min(), x.max())
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[:3]:
        ax.set_ylim(-0.03, 1.03)
    fig.suptitle(f'Modelo de Ising con campo transverso — cadena cerrada\nDiagonalización exacta, $N={n_q}$', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()

# Celda 9 · nombre original: comparar_tamanos_mz2
def comparar_tamanos_mz2(tamanos=(4, 6, 8), puntos: int=81):
    plt.figure(figsize=(8, 5))
    for n_q in tamanos:
        datos = barrido_observables_ising_cerrado(n_q=n_q, J=1.0, h_min=0.0, h_max=2.0, puntos=puntos)
        plt.plot(datos['h_sobre_J'], datos['mz2'], linewidth=2, label=f'$N={n_q}$')
    plt.axvline(1.0, linestyle='--', linewidth=1.5, label='$h/J=1$')
    plt.xlabel('$h/J$')
    plt.ylabel('$\\langle M_z^2\\rangle$')
    plt.title('Efecto del tamaño del sistema — cadena cerrada')
    plt.xlim(0, 2)
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def circuito_tfim_trotter_cerrado(
    J: float,
    h: float,
    n_q: int,
    r: int,
    t: float
) -> QuantumCircuit:
    """
    Construye el circuito de evolución Trotter de primer orden
    para el modelo de Ising con campo transverso con frontera periódica.

    Hamiltoniano:
        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    con Z_N = Z_0.

    El estado inicial es |00...0>.
    """

    if n_q < 3:
        raise ValueError(
            "Para una cadena periódica sin enlaces duplicados, "
            "n_q debe ser al menos 3."
        )

    if r < 1:
        raise ValueError("r debe ser al menos 1.")

    if t < 0:
        raise ValueError("t debe ser mayor o igual que cero.")

    dt = t / r

    qc = QuantumCircuit(n_q)

    for _ in range(r):

        # Interacciones ZZ de la cadena cerrada
        for i in range(n_q):
            j = (i + 1) % n_q

            qc.rzz(
                -2.0 * J * dt,
                i,
                j
            )

        # Campo transversal
        for i in range(n_q):
            qc.rx(
                -2.0 * h * dt,
                i
            )

    return qc

def evolucion_tfim_trotter_cerrada(
    J: float,
    h: float,
    n_q: int,
    r: int,
    t: float
) -> Statevector:
    """
    Devuelve el estado evolucionado mediante Trotter de primer orden.
    """

    qc = circuito_tfim_trotter_cerrado(
        J=J,
        h=h,
        n_q=n_q,
        r=r,
        t=t
    )

    return Statevector.from_instruction(qc)

def magnetizacion_z(n_q: int) -> SparsePauliOp:
    r"""
    Magnetización longitudinal promedio:

        M_z = (1/N) Σ_i Z_i.
    """

    if n_q < 1:
        raise ValueError("n_q debe ser al menos 1.")

    terminos = [
        ("Z", [i], 1.0 / n_q)
        for i in range(n_q)
    ]

    return SparsePauliOp.from_sparse_list(
        terminos,
        num_qubits=n_q
    )

def barrido_temporal_tfim_trotter_cerrado(
    J: float,
    h: float,
    n_q: int,
    r: int,
    t_min: float = 0.0,
    t_max: float = 5.0,
    puntos: int = 101
) -> dict:
    """
    Calcula la dinámica temporal trotterizada del TFIM cerrado.

    El estado inicial es |00...0>.

    Devuelve:
        tiempos
        <Mz>(t)
        <Czz>(t)
    """

    if n_q < 3:
        raise ValueError(
            "Para una cadena cerrada se requiere n_q >= 3."
        )

    if r < 1:
        raise ValueError("r debe ser al menos 1.")

    if t_min < 0:
        raise ValueError("t_min debe ser mayor o igual que cero.")

    if t_max < t_min:
        raise ValueError("t_max debe ser mayor o igual que t_min.")

    if puntos < 2:
        raise ValueError("puntos debe ser al menos 2.")

    tiempos = np.linspace(
        t_min,
        t_max,
        puntos
    )

    Mz = magnetizacion_z(n_q)

    Czz = correlacion_zz_vecinos_cerrada(n_q)

    valores_mz = np.empty(puntos)
    valores_czz = np.empty(puntos)

    for k, t in enumerate(tiempos):

        psi_t = evolucion_tfim_trotter_cerrada(
            J=J,
            h=h,
            n_q=n_q,
            r=r,
            t=float(t)
        )

        valores_mz[k] = float(
            np.real_if_close(
                psi_t.expectation_value(Mz)
            )
        )

        valores_czz[k] = float(
            np.real_if_close(
                psi_t.expectation_value(Czz)
            )
        )

    return {
        "tiempos": tiempos,
        "mz": valores_mz,
        "czz": valores_czz,
        "J": J,
        "h": h,
        "n_q": n_q,
        "r": r
    }


# ======================================================================================
# VQE CON PENNYLANE
# Funciones trasladadas desde "Implementacion del sistema.ipynb".
# ======================================================================================

def _dependencias_vqe():
    """Importa las dependencias opcionales sólo cuando se ejecuta el VQE."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as error:
        raise ImportError(
            "Las funciones VQE requieren PennyLane y sus dependencias."
        ) from error
    return qml, pnp


def hamiltoniano_ising_pennylane(
    n_qubits: int,
    J: float,
    h: float,
):
    """
    Hamiltoniano TFIM periódico para PennyLane:

        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i.

    Se usa un nombre específico para no reemplazar a ``hamiltoniano_ising``,
    que construye el operador equivalente con Qiskit.
    """
    qml, _ = _dependencias_vqe()

    if n_qubits < 2:
        raise ValueError("El sistema debe tener al menos 2 qubits.")

    enlaces = (
        [(0, 1)]
        if n_qubits == 2
        else [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    )

    coeficientes = []
    operadores = []

    for i, j in enlaces:
        coeficientes.append(-J)
        operadores.append(qml.PauliZ(i) @ qml.PauliZ(j))

    for i in range(n_qubits):
        coeficientes.append(-h)
        operadores.append(qml.PauliX(i))

    return qml.dot(coeficientes, operadores)


def bloque_rotaciones(parametros_bloque, n_qubits: int) -> None:
    """Aplica RX, RY y RZ sobre cada qubit."""
    qml, _ = _dependencias_vqe()

    for q in range(n_qubits):
        qml.RX(parametros_bloque[q, 0], wires=q)
        qml.RY(parametros_bloque[q, 1], wires=q)
        qml.RZ(parametros_bloque[q, 2], wires=q)


def anillo_cnots(n_qubits: int) -> None:
    """Entrelaza vecinos; para dos qubits usa una sola CNOT."""
    qml, _ = _dependencias_vqe()

    if n_qubits == 2:
        qml.CNOT(wires=[0, 1])
        return

    for control in range(n_qubits):
        objetivo = (control + 1) % n_qubits
        qml.CNOT(wires=[control, objetivo])


def ansatz_ising(parametros, n_qubits: int, n_layers: int) -> None:
    """Ansatz de rotaciones locales y anillos de CNOT."""
    bloque_rotaciones(parametros[0], n_qubits)

    for layer in range(n_layers):
        anillo_cnots(n_qubits)
        bloque_rotaciones(parametros[layer + 1], n_qubits)


def vqe_ising(
    n_qubits: int,
    J: float,
    h: float,
    n_layers: int = 1,
    learning_rate: float = 0.05,
    max_steps: int = 500,
    tol: float = 1e-8,
    patience: int = 15,
    seed: int = 42,
    mostrar_cada: int = 25,
    calcular_exacto: bool = True,
    parametros_iniciales=None,
    verbose: bool = True,
):
    """Ejecuta un VQE del TFIM con ADAM y simulación exacta sin shots."""
    qml, pnp = _dependencias_vqe()

    if n_qubits < 2:
        raise ValueError("n_qubits debe ser al menos 2.")
    if n_layers < 1:
        raise ValueError("n_layers debe ser al menos 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo.")

    H = hamiltoniano_ising_pennylane(
        n_qubits=n_qubits,
        J=J,
        h=h,
    )
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_energia(parametros):
        ansatz_ising(parametros, n_qubits=n_qubits, n_layers=n_layers)
        return qml.expval(H)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuito_estado(parametros):
        ansatz_ising(parametros, n_qubits=n_qubits, n_layers=n_layers)
        return qml.state()

    if parametros_iniciales is None:
        rng = np.random.default_rng(seed)
        parametros = pnp.array(
            rng.uniform(
                low=-0.1,
                high=0.1,
                size=(n_layers + 1, n_qubits, 3),
            ),
            requires_grad=True,
        )
    else:
        parametros = pnp.array(
            np.asarray(parametros_iniciales),
            requires_grad=True,
        )
    optimizador = qml.AdamOptimizer(stepsize=learning_rate)

    historial = [float(circuito_energia(parametros))]
    pasos_estables = 0

    if verbose:
        print(f"Energía inicial: {historial[0]:.12f}")

    for step in range(1, max_steps + 1):
        parametros = optimizador.step(circuito_energia, parametros)
        energia_actual = float(circuito_energia(parametros))
        historial.append(energia_actual)
        diferencia = abs(historial[-1] - historial[-2])

        if verbose and (step % mostrar_cada == 0 or step == 1):
            print(
                f"Paso {step:4d} | "
                f"E = {energia_actual:.12f} | "
                f"ΔE = {diferencia:.3e}"
            )

        pasos_estables = pasos_estables + 1 if diferencia < tol else 0
        if pasos_estables >= patience:
            if verbose:
                print(f"\nConvergencia alcanzada en el paso {step}.")
            break

    energia_vqe = float(circuito_energia(parametros))
    resultado = {
        "energia_vqe": energia_vqe,
        "parametros_optimos": parametros,
        "historial": np.asarray(historial),
        "estado_vqe": np.asarray(circuito_estado(parametros)),
        "hamiltoniano": H,
        "circuito_energia": circuito_energia,
        "circuito_estado": circuito_estado,
        "pasos_realizados": len(historial) - 1,
    }

    if calcular_exacto and n_qubits <= 12:
        matriz_H = qml.matrix(H, wire_order=range(n_qubits))
        energia_exacta = float(np.linalg.eigvalsh(np.asarray(matriz_H))[0])
        resultado["energia_exacta"] = energia_exacta
        resultado["error_absoluto"] = abs(energia_vqe - energia_exacta)

        if verbose:
            print("\nResultado final")
            print("------------------------------")
            print(f"Energía VQE:    {energia_vqe:.12f}")
            print(f"Energía exacta: {energia_exacta:.12f}")
            print(f"Error absoluto: {resultado['error_absoluto']:.3e}")
    elif calcular_exacto and verbose:
        print("\nNo se realizó diagonalización exacta: n_qubits > 12.")

    return resultado


def matriz_operador(operador, n_qubits: int) -> np.ndarray:
    """Regresa la matriz completa de un operador PennyLane."""
    qml, _ = _dependencias_vqe()
    return np.asarray(
        qml.matrix(operador, wire_order=range(n_qubits)),
        dtype=complex,
    )


def valor_esperado_estado(
    estado: np.ndarray,
    operador_matriz: np.ndarray,
) -> float:
    """Calcula <psi|O|psi>."""
    estado = np.asarray(estado, dtype=complex)
    return float(np.real(np.vdot(estado, operador_matriz @ estado)))


def matrices_observables_tfim(n_qubits: int) -> dict[str, np.ndarray]:
    """Construye matrices de Mz, Mz², Mx y Czz con frontera periódica."""
    qml, _ = _dependencias_vqe()

    if n_qubits < 2:
        raise ValueError("Se necesitan al menos 2 qubits.")

    dimension = 2 ** n_qubits
    Mz = np.zeros((dimension, dimension), dtype=complex)
    Mx = np.zeros((dimension, dimension), dtype=complex)

    for i in range(n_qubits):
        Mz += matriz_operador(qml.PauliZ(i), n_qubits)
        Mx += matriz_operador(qml.PauliX(i), n_qubits)

    Mz /= n_qubits
    Mx /= n_qubits
    Mz2 = Mz @ Mz

    parejas = (
        [(0, 1)]
        if n_qubits == 2
        else [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    )
    Czz = np.zeros((dimension, dimension), dtype=complex)

    for i, j in parejas:
        Czz += matriz_operador(qml.PauliZ(i) @ qml.PauliZ(j), n_qubits)
    Czz /= len(parejas)

    return {"Mz": Mz, "Mz2": Mz2, "Mx": Mx, "Czz": Czz}


def medir_observables_tfim(
    estado: np.ndarray,
    n_qubits: int,
) -> dict[str, float]:
    """Mide Mz, Mz², Mx y Czz sobre un vector de estado."""
    operadores = matrices_observables_tfim(n_qubits)
    return {
        nombre: valor_esperado_estado(estado, matriz)
        for nombre, matriz in operadores.items()
    }


def estado_base_exacto(H, n_qubits: int) -> tuple[float, np.ndarray]:
    """Diagonaliza exactamente un Hamiltoniano PennyLane."""
    qml, _ = _dependencias_vqe()
    matriz_H = np.asarray(
        qml.matrix(H, wire_order=range(n_qubits)),
        dtype=complex,
    )
    energias, estados = np.linalg.eigh(matriz_H)
    return float(np.real(energias[0])), estados[:, 0]


def mejor_vqe_ising(
    n_qubits: int,
    J: float,
    h: float,
    n_layers: int = 3,
    learning_rate: float = 0.03,
    max_steps: int = 800,
    seeds: tuple[int, ...] = (3, 7, 11, 19, 42),
):
    """Ejecuta varios reinicios y conserva el VQE de menor energía."""
    mejor_resultado = None

    for seed in seeds:
        resultado = vqe_ising(
            n_qubits=n_qubits,
            J=J,
            h=h,
            n_layers=n_layers,
            learning_rate=learning_rate,
            max_steps=max_steps,
            tol=1e-9,
            patience=25,
            seed=seed,
            mostrar_cada=max_steps + 1,
            calcular_exacto=False,
        )

        if (
            mejor_resultado is None
            or resultado["energia_vqe"] < mejor_resultado["energia_vqe"]
        ):
            mejor_resultado = resultado

    return mejor_resultado


def comparar_vqe_tfim(
    n_qubits: int = 4,
    J: float = 1.0,
    valores_h: tuple[float, ...] = (0.5, 1.0, 2.0),
    n_layers: int = 3,
    learning_rate: float = 0.03,
    max_steps: int = 800,
    seeds: tuple[int, ...] = (3, 7, 11, 19, 42),
):
    """Compara VQE y diagonalización exacta para varios campos."""
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "comparar_vqe_tfim requiere pandas para construir la tabla."
        ) from error

    filas = []
    estados = {}

    for h in valores_h:
        print("\n" + "=" * 60)
        print(f"J = {J}, h = {h}, h/J = {h / J}")
        print("=" * 60)

        resultado_vqe = mejor_vqe_ising(
            n_qubits=n_qubits,
            J=J,
            h=h,
            n_layers=n_layers,
            learning_rate=learning_rate,
            max_steps=max_steps,
            seeds=seeds,
        )
        H = resultado_vqe["hamiltoniano"]
        estado_vqe = np.asarray(resultado_vqe["estado_vqe"], dtype=complex)
        energia_vqe = float(resultado_vqe["energia_vqe"])
        energia_exacta, estado_exacto = estado_base_exacto(H, n_qubits)
        obs_vqe = medir_observables_tfim(estado_vqe, n_qubits)
        obs_exactos = medir_observables_tfim(estado_exacto, n_qubits)
        fidelidad = float(np.abs(np.vdot(estado_exacto, estado_vqe)) ** 2)

        filas.append({
            "h": h,
            "h/J": h / J,
            "E_VQE": energia_vqe,
            "E_exacta": energia_exacta,
            "error_E": abs(energia_vqe - energia_exacta),
            "fidelidad": fidelidad,
            "Mz_VQE": obs_vqe["Mz"],
            "Mz_exacto": obs_exactos["Mz"],
            "Mz2_VQE": obs_vqe["Mz2"],
            "Mz2_exacto": obs_exactos["Mz2"],
            "Mx_VQE": obs_vqe["Mx"],
            "Mx_exacto": obs_exactos["Mx"],
            "Czz_VQE": obs_vqe["Czz"],
            "Czz_exacto": obs_exactos["Czz"],
        })
        estados[h] = {
            "estado_vqe": estado_vqe,
            "estado_exacto": estado_exacto,
            "parametros": resultado_vqe["parametros_optimos"],
            "resultado_vqe": resultado_vqe,
        }

    return pd.DataFrame(filas), estados


def graficar_observables_vqe(tabla) -> None:
    """Grafica los observables VQE frente a la solución exacta."""
    comparaciones = [
        ("Mz_VQE", "Mz_exacto", r"$\langle M_z \rangle$"),
        ("Mz2_VQE", "Mz2_exacto", r"$\langle M_z^2 \rangle$"),
        ("Mx_VQE", "Mx_exacto", r"$\langle M_x \rangle$"),
        ("Czz_VQE", "Czz_exacto", r"$C_{ZZ}$"),
    ]

    for columna_vqe, columna_exacta, ylabel in comparaciones:
        plt.figure(figsize=(7, 4))
        plt.plot(
            tabla["h/J"],
            tabla[columna_vqe],
            marker="o",
            label="VQE",
        )
        plt.plot(
            tabla["h/J"],
            tabla[columna_exacta],
            marker="s",
            linestyle="--",
            label="Exacto",
        )
        plt.axvline(1.0, linestyle=":", label=r"$h/J=1$")
        plt.xlabel(r"$h/J$")
        plt.ylabel(ylabel)
        plt.title(f"Comparación VQE vs exacta: {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()


# ======================================================================================
# FLUJO REPRODUCIBLE DEL CHALLENGE 3
# ======================================================================================


def estilo_graficas_tfim() -> None:
    """Configura un estilo compacto y consistente para el notebook."""
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


def observables_tfim_estado(
    estado: Statevector,
    periodic: bool = True,
) -> dict[str, float]:
    """Calcula Mz, Mz², Mx y la correlación ZZ media."""
    n_q = estado.num_qubits
    operadores = {
        "mz": magnetizacion_z(n_q),
        "mz2": magnetizacion_z_cuadrada(n_q),
        "mx": magnetizacion_x(n_q),
        "czz": correlacion_zz_vecinos(n_q, periodic=periodic),
    }
    return {
        nombre: float(np.real_if_close(estado.expectation_value(operador)))
        for nombre, operador in operadores.items()
    }


def fidelidad_estados(
    estado_a: Statevector,
    estado_b: Statevector,
) -> float:
    """Fidelidad pura |<a|b>|²."""
    return float(abs(np.vdot(estado_a.data, estado_b.data)) ** 2)


def convergencia_trotter_tfim(
    valores_r,
    J: float = 1.0,
    h: float = 1.0,
    n_q: int = 6,
    t: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compara la evolución Trotter contra ED al variar los pasos r."""
    exacto = evolve_tfim_exact(
        t=t, h=h, J=J, n_qubits=n_q, r=1, periodic=True
    )
    obs_exacto = observables_tfim_estado(exacto, periodic=True)
    fidelidades = []
    errores_mz = []
    errores_czz = []

    for r in valores_r:
        trotter = evolucion_tfim_trotter_cerrada(
            J=J, h=h, n_q=n_q, r=int(r), t=t
        )
        obs_trotter = observables_tfim_estado(trotter, periodic=True)
        fidelidades.append(fidelidad_estados(exacto, trotter))
        errores_mz.append(abs(obs_trotter["mz"] - obs_exacto["mz"]))
        errores_czz.append(abs(obs_trotter["czz"] - obs_exacto["czz"]))

    return {
        "r": np.asarray(valores_r, dtype=int),
        "dt": t / np.asarray(valores_r, dtype=float),
        "fidelidad": np.asarray(fidelidades),
        "error_mz": np.asarray(errores_mz),
        "error_czz": np.asarray(errores_czz),
    }


def escalado_trotter_tfim(
    tamanos=(4, 6, 8, 10, 12),
    J: float = 1.0,
    h: float = 1.0,
    r: int = 4,
    t: float = 1.0,
) -> list[dict]:
    """Evalúa costo y autoconsistencia r frente a 2r para varios tamaños."""
    resultados = []
    for n_q in tamanos:
        circuito_r = circuito_tfim_trotter_cerrado(
            J=J, h=h, n_q=int(n_q), r=r, t=t
        )
        circuito_2r = circuito_tfim_trotter_cerrado(
            J=J, h=h, n_q=int(n_q), r=2 * r, t=t
        )
        inicio = perf_counter()
        estado_r = Statevector.from_instruction(circuito_r)
        estado_2r = Statevector.from_instruction(circuito_2r)
        duracion = perf_counter() - inicio
        resultados.append(
            {
                "n_q": int(n_q),
                "profundidad": circuito_r.depth(),
                "compuertas": circuito_r.size(),
                "fidelidad_r_2r": fidelidad_estados(
                    estado_r, estado_2r
                ),
                "tiempo_s": duracion,
                "memoria_estado_mb": 16 * (2**int(n_q)) / 2**20,
            }
        )
        del estado_r, estado_2r
    return resultados


def dinamica_tfim_comparada(
    J: float,
    h: float,
    n_q: int,
    r: int,
    tiempos,
) -> dict[str, np.ndarray]:
    """Dinámica Trotter y ED con observables y fidelidad en cada tiempo."""
    datos = {
        "tiempos": np.asarray(tiempos, dtype=float),
        "mz_trotter": [],
        "mz_exacta": [],
        "czz_trotter": [],
        "czz_exacta": [],
        "fidelidad": [],
    }
    for tiempo in datos["tiempos"]:
        trotter = evolucion_tfim_trotter_cerrada(
            J=J, h=h, n_q=n_q, r=r, t=float(tiempo)
        )
        exacta = evolve_tfim_exact(
            t=float(tiempo),
            h=h,
            J=J,
            n_qubits=n_q,
            r=1,
            periodic=True,
        )
        obs_trotter = observables_tfim_estado(trotter, periodic=True)
        obs_exacta = observables_tfim_estado(exacta, periodic=True)
        datos["mz_trotter"].append(obs_trotter["mz"])
        datos["mz_exacta"].append(obs_exacta["mz"])
        datos["czz_trotter"].append(obs_trotter["czz"])
        datos["czz_exacta"].append(obs_exacta["czz"])
        datos["fidelidad"].append(fidelidad_estados(trotter, exacta))
    for clave in datos:
        datos[clave] = np.asarray(datos[clave])
    return datos


def barrido_fase_ed_tfim(
    valores_h_sobre_J=(0.5, 1.0, 2.0),
    J: float = 1.0,
    n_q: int = 6,
) -> list[dict]:
    """Observables del estado fundamental exacto para campos seleccionados."""
    filas = []
    for razon in valores_h_sobre_J:
        energia, estado = estado_base_ising_cerrado(
            J=J,
            h=float(razon) * J,
            n_q=n_q,
            devolver_energia=True,
        )
        obs = observables_tfim_estado(estado, periodic=True)
        filas.append(
            {
                "h/J": float(razon),
                "E0/N": energia / n_q,
                "Mz": obs["mz"],
                "sqrt_Mz2": np.sqrt(max(obs["mz2"], 0.0)),
                "Mx": obs["mx"],
                "Czz": obs["czz"],
            }
        )
    return filas


def barrido_vqe_tfim_unico(
    valores_h_sobre_J=(0.5, 1.0, 2.0),
    J: float = 1.0,
    n_layers: int = 3,
    learning_rate: float = 0.04,
    max_steps: int = 500,
    seed: int = 17,
) -> tuple[list[dict], dict]:
    """Ejecuta exactamente un VQE de 6 espines por cada valor de h/J."""
    n_qubits = 6
    filas = []
    detalles = {}
    for razon in valores_h_sobre_J:
        h = float(razon) * J
        rng = np.random.default_rng(seed)
        parametros_iniciales = rng.normal(
            0.0, 0.03, size=(n_layers + 1, n_qubits, 3)
        )
        angulo_producto = np.arcsin(min(1.0, abs(h) / (2.0 * abs(J))))
        parametros_iniciales[0, :, 0] = 0.0
        parametros_iniciales[0, :, 1] = angulo_producto
        parametros_iniciales[0, :, 2] = 0.0
        resultado = vqe_ising(
            n_qubits=n_qubits,
            J=J,
            h=h,
            n_layers=n_layers,
            learning_rate=learning_rate,
            max_steps=max_steps,
            tol=1e-8,
            patience=25,
            seed=seed,
            mostrar_cada=max_steps + 1,
            calcular_exacto=False,
            parametros_iniciales=parametros_iniciales,
            verbose=False,
        )
        qml, _ = _dependencias_vqe()
        matriz_h = np.asarray(
            qml.matrix(
                resultado["hamiltoniano"],
                wire_order=range(n_qubits),
            ),
            dtype=complex,
        )
        energias_ed, estados_ed = np.linalg.eigh(matriz_h)
        energia_exacta = float(energias_ed[0])
        estado_exacto = estados_ed[:, 0]
        estado_vqe = np.asarray(resultado["estado_vqe"], dtype=complex)
        obs_vqe = medir_observables_tfim(estado_vqe, n_qubits)
        obs_exacto = medir_observables_tfim(estado_exacto, n_qubits)
        fidelidad = float(abs(np.vdot(estado_exacto, estado_vqe)) ** 2)
        fidelidad_subespacio_2 = float(
            np.sum(
                np.abs(estados_ed[:, :2].conj().T @ estado_vqe) ** 2
            )
        )
        fila = {
            "h/J": float(razon),
            "E_VQE/N": resultado["energia_vqe"] / n_qubits,
            "E_ED/N": energia_exacta / n_qubits,
            "error_E": abs(resultado["energia_vqe"] - energia_exacta),
            "error_E_por_spin": abs(
                resultado["energia_vqe"] - energia_exacta
            ) / n_qubits,
            "error_E_relativo_pct": 100.0
            * abs(resultado["energia_vqe"] - energia_exacta)
            / abs(energia_exacta),
            "fidelidad": fidelidad,
            "fidelidad_subespacio_2": fidelidad_subespacio_2,
            "infidelidad_F0": 1.0 - fidelidad,
            "infidelidad_subespacio_2": 1.0 - fidelidad_subespacio_2,
            "gap_ED": float(energias_ed[1] - energias_ed[0]),
            "Mz2_VQE": obs_vqe["Mz2"],
            "Mz2_ED": obs_exacto["Mz2"],
            "error_Mz2": abs(obs_vqe["Mz2"] - obs_exacto["Mz2"]),
            "Mx_VQE": obs_vqe["Mx"],
            "Mx_ED": obs_exacto["Mx"],
            "error_Mx": abs(obs_vqe["Mx"] - obs_exacto["Mx"]),
            "Czz_VQE": obs_vqe["Czz"],
            "Czz_ED": obs_exacto["Czz"],
            "error_Czz": abs(obs_vqe["Czz"] - obs_exacto["Czz"]),
            "pasos": resultado["pasos_realizados"],
        }
        filas.append(fila)
        detalles[float(razon)] = resultado
    return filas, detalles

