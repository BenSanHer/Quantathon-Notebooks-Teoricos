"""
Funciones extraídas de los cuatro notebooks del proyecto TFIM/Ising.

Las funciones con nombres repetidos fueron renombradas para que ninguna
definición se pierda al importarlas desde este módulo.
"""

from __future__ import annotations

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

