"""Interfaz principal y utilidades de presentación para ``Resultados.ipynb``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from funciones_qiskit import *  # noqa: F401,F403 - fachada deliberada


COLORES = {
    "exacta": "#16324F",
    "trotter": "#E07A5F",
    "critico": "#7A5195",
    "ferro": "#2A9D8F",
    "paramagnetico": "#F4A261",
    "gris": "#6C757D",
}


def estilo_resultados() -> None:
    """Configura Matplotlib con un estilo coherente para todo el notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (8.4, 4.8),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def finalizar_ejes(
    ax,
    titulo: str,
    xlabel: str,
    ylabel: str,
    leyenda: bool = True,
) -> None:
    """Aplica títulos, etiquetas y leyenda sin repetir sintaxis."""
    ax.set_title(titulo, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if leyenda:
        ax.legend()


def graficar_linea_base(datos: Mapping[str, np.ndarray], n_q: int):
    """Panel educativo de observables del estado base obtenidos por ED."""
    x = datos["h_sobre_J"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), sharex=True)
    paneles = (
        ("mz2", r"$\langle M_z^2\rangle$", "Orden longitudinal"),
        ("mx", r"$\langle M_x\rangle$", "Respuesta transversal"),
        ("czz", r"$\langle C_{zz}\rangle$", "Correlación vecina"),
    )
    for ax, (clave, ylabel, titulo) in zip(axes, paneles):
        ax.plot(x, datos[clave], color=COLORES["exacta"], label=f"ED, N={n_q}")
        ax.axvline(1.0, color=COLORES["critico"], ls="--", label=r"$h/J=1$")
        finalizar_ejes(ax, titulo, r"$h/J$", ylabel)
    fig.suptitle("Línea base clásica del TFIM", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def graficar_dinamica_comparada(
    resultados: Mapping[str, Mapping[str, np.ndarray]],
    observable: str,
    ylabel: str,
):
    """Compara ED y Trotter para los tres regímenes del reto."""
    fig, axes = plt.subplots(1, len(resultados), figsize=(14, 4), sharey=True)
    for ax, (etiqueta, datos) in zip(np.atleast_1d(axes), resultados.items()):
        ax.plot(
            datos["tiempos"],
            datos[f"{observable}_exacta"],
            color=COLORES["exacta"],
            label="ED",
        )
        ax.plot(
            datos["tiempos"],
            datos[f"{observable}_trotter"],
            color=COLORES["trotter"],
            ls="--",
            label=f"Trotter, r={datos['r']}",
        )
        finalizar_ejes(ax, etiqueta, r"$Jt$", ylabel)
    fig.suptitle("Dinámica desde $|00\\cdots0\\rangle$", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def graficar_convergencia(datos: Mapping[str, np.ndarray]):
    """Muestra infidelidad y error máximo de observables frente a Δt."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].loglog(
        datos["dt"],
        datos["infidelidad"],
        "o-",
        color=COLORES["trotter"],
    )
    finalizar_ejes(
        axes[0], "Error de estado", r"$\Delta t$", r"$1-F$", leyenda=False
    )
    axes[1].loglog(
        datos["dt"],
        datos["error_max_observable"],
        "o-",
        color=COLORES["critico"],
    )
    finalizar_ejes(
        axes[1],
        "Peor error observable",
        r"$\Delta t$",
        "error absoluto",
        leyenda=False,
    )
    fig.suptitle("Convergencia de Trotter de primer orden", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def graficar_escalado(datos: Mapping[str, np.ndarray]):
    """Relaciona calidad del estado y costo del circuito con N."""
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.plot(
        datos["n_q"],
        datos["fidelidad"],
        "o-",
        color=COLORES["exacta"],
        label="Fidelidad",
    )
    ax.set_ylim(0.9, 1.005)
    finalizar_ejes(ax, "Calidad vs. tamaño", "qubits N", "fidelidad")
    ax2 = ax.twinx()
    ax2.plot(
        datos["n_q"],
        datos["compuertas"],
        "s--",
        color=COLORES["trotter"],
        label="compuertas",
    )
    ax2.set_ylabel("número de compuertas")
    ax2.spines["right"].set_visible(True)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return fig, (ax, ax2)


def graficar_fermi_hubbard(datos: Mapping[str, np.ndarray]):
    """Presenta densidad, doble ocupación y conservación de partículas."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for sitio, serie in enumerate(datos["densidades"].T):
        axes[0].plot(datos["tiempos"], serie, label=f"sitio {sitio}")
    finalizar_ejes(
        axes[0], "Densidad local", r"$t$", r"$\langle n_i\rangle$"
    )
    axes[1].plot(
        datos["tiempos"],
        datos["doble_ocupacion"],
        color=COLORES["critico"],
    )
    finalizar_ejes(
        axes[1],
        "Doble ocupación",
        r"$t$",
        r"$\langle n_{\uparrow}n_{\downarrow}\rangle$",
        leyenda=False,
    )
    axes[2].plot(
        datos["tiempos"],
        datos["numero_particulas"],
        color=COLORES["ferro"],
    )
    finalizar_ejes(
        axes[2],
        "Chequeo de conservación",
        r"$t$",
        r"$\langle N\rangle$",
        leyenda=False,
    )
    fig.suptitle("Extensión opcional: Fermi-Hubbard 1D", fontweight="bold")
    fig.tight_layout()
    return fig, axes


def tabla_markdown(
    columnas: Sequence[str],
    filas: Sequence[Sequence[object]],
    decimales: int = 5,
) -> str:
    """Convierte resultados pequeños en una tabla Markdown legible."""
    def formato(valor):
        if isinstance(valor, (float, np.floating)):
            return f"{valor:.{decimales}f}"
        return str(valor)

    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join("---" for _ in columnas) + " |",
    ]
    lineas.extend(
        "| " + " | ".join(formato(valor) for valor in fila) + " |"
        for fila in filas
    )
    return "\n".join(lineas)

