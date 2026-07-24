"""Prototipo VQE opcional con Guppy rescatado de ``VQE-Guppy.ipynb``.

La API de Guppy se importa sólo al construir el circuito para que el resto del
proyecto siga siendo ejecutable sin esa dependencia.
"""

from __future__ import annotations


def valor_esperado_conteos(conteos, paulis: str) -> float:
    """Estima el valor esperado de una cadena de Pauli desde conteos."""
    total = sum(conteos.values())
    if total == 0:
        raise ValueError("Los conteos están vacíos.")
    esperado = 0.0
    for medicion, cuenta in conteos.items():
        autovalor = 1
        for i, pauli in enumerate(paulis):
            if pauli != "I":
                autovalor *= (-1) ** int(medicion[i])
        esperado += autovalor * cuenta / total
    return esperado


def crear_circuito_vqe_guppy(thetas, paulis: str):
    """Construye el ansatz de dos qubits usado en el prototipo original."""
    try:
        from guppylang import guppy
        from guppylang.std.angles import angle
        from guppylang.std.builtins import array, comptime, result
        from guppylang.std.quantum import (
            cx,
            h,
            measure_array,
            qubit,
            ry,
            sdg,
        )
    except ImportError as error:
        raise ImportError(
            "Guppy no está instalado. Ejecuta este módulo en el entorno de "
            "Quantinuum correspondiente."
        ) from error

    if len(thetas) != 2 or len(paulis) != 2:
        raise ValueError("El prototipo actual usa exactamente dos qubits.")
    codigo = [{"X": 0, "Y": 1, "Z": 2, "I": 3}[p] for p in paulis]
    angulos = [float(theta) for theta in thetas]

    @guppy
    def circuito() -> None:
        qs = array(qubit() for _ in range(comptime(2)))
        ry(qs[0], angle(comptime(angulos[0])))
        ry(qs[1], angle(comptime(angulos[1])))
        cx(qs[0], qs[1])
        i = 0
        for base in comptime(codigo):
            if base == 0:
                h(qs[i])
            elif base == 1:
                sdg(qs[i])
                h(qs[i])
            i += 1
        result("m", measure_array(qs))

    return circuito


def energia_vqe_guppy(
    thetas,
    hamiltoniano=(("ZI", 1.0), ("IZ", 1.0), ("XX", 1.0)),
    n_shots: int = 4000,
) -> float:
    """Evalúa la energía del prototipo mediante el emulador Guppy."""
    energia = 0.0
    for paulis, coeficiente in hamiltoniano:
        circuito = crear_circuito_vqe_guppy(thetas, paulis)
        ejecucion = circuito.emulator(n_qubits=2).with_shots(n_shots).run()
        conteos = ejecucion.register_counts()["m"]
        energia += coeficiente * valor_esperado_conteos(conteos, paulis)
    return energia

