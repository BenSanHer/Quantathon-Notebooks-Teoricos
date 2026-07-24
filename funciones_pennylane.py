"""VQE opcional con PennyLane rescatado de los notebooks de implementación.

PennyLane se importa de forma diferida: el módulo puede existir en el
repositorio sin convertirse en una dependencia del notebook principal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _qml():
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as error:
        raise ImportError(
            "Instala PennyLane o ejecuta este módulo en el entorno "
            "'pennylane-env'."
        ) from error
    return qml, pnp


def hamiltoniano_tfim_pennylane(n_qubits: int, J: float, h: float):
    qml, _ = _qml()
    if n_qubits < 2:
        raise ValueError("Se necesitan al menos dos qubits.")
    enlaces = (
        [(0, 1)]
        if n_qubits == 2
        else [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    )
    coeficientes = [-J] * len(enlaces) + [-h] * n_qubits
    operadores = [
        qml.PauliZ(i) @ qml.PauliZ(j) for i, j in enlaces
    ]
    operadores.extend(qml.PauliX(i) for i in range(n_qubits))
    return qml.dot(coeficientes, operadores)


def ansatz_tfim_pennylane(parametros, n_qubits: int, n_layers: int) -> None:
    qml, _ = _qml()
    for capa in range(n_layers + 1):
        for q in range(n_qubits):
            qml.RX(parametros[capa, q, 0], wires=q)
            qml.RY(parametros[capa, q, 1], wires=q)
            qml.RZ(parametros[capa, q, 2], wires=q)
        if capa < n_layers:
            if n_qubits == 2:
                qml.CNOT(wires=[0, 1])
            else:
                for control in range(n_qubits):
                    qml.CNOT(wires=[control, (control + 1) % n_qubits])


def vqe_tfim_pennylane(
    n_qubits: int,
    J: float,
    h: float,
    n_layers: int = 2,
    learning_rate: float = 0.04,
    max_steps: int = 400,
    seed: int = 42,
) -> dict:
    """Ejecuta un VQE sin shots y devuelve energía, estado e historial."""
    qml, pnp = _qml()
    hamiltoniano = hamiltoniano_tfim_pennylane(n_qubits, J, h)
    dispositivo = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dispositivo, interface="autograd")
    def energia(parametros):
        ansatz_tfim_pennylane(parametros, n_qubits, n_layers)
        return qml.expval(hamiltoniano)

    @qml.qnode(dispositivo, interface="autograd")
    def estado(parametros):
        ansatz_tfim_pennylane(parametros, n_qubits, n_layers)
        return qml.state()

    rng = np.random.default_rng(seed)
    parametros = pnp.array(
        rng.uniform(-0.1, 0.1, (n_layers + 1, n_qubits, 3)),
        requires_grad=True,
    )
    optimizador = qml.AdamOptimizer(stepsize=learning_rate)
    historial = [float(energia(parametros))]
    for _ in range(max_steps):
        parametros = optimizador.step(energia, parametros)
        historial.append(float(energia(parametros)))
    matriz = np.asarray(
        qml.matrix(hamiltoniano, wire_order=range(n_qubits)),
        dtype=complex,
    )
    energia_exacta = float(np.linalg.eigvalsh(matriz)[0])
    return {
        "energia_vqe": historial[-1],
        "energia_exacta": energia_exacta,
        "error_absoluto": abs(historial[-1] - energia_exacta),
        "estado_vqe": np.asarray(estado(parametros)),
        "parametros_optimos": parametros,
        "historial": np.asarray(historial),
    }


def comparar_vqe_tfim(
    valores_h: Sequence[float] = (0.5, 1.0, 2.0),
    n_qubits: int = 4,
    J: float = 1.0,
    **opciones,
) -> list[dict]:
    """Ejecuta VQE para varios campos y devuelve registros simples."""
    resultados = []
    for h in valores_h:
        resultado = vqe_tfim_pennylane(n_qubits, J, float(h), **opciones)
        resultados.append(
            {
                "h_sobre_J": float(h) / J,
                "energia_vqe": resultado["energia_vqe"],
                "energia_exacta": resultado["energia_exacta"],
                "error_absoluto": resultado["error_absoluto"],
            }
        )
    return resultados

