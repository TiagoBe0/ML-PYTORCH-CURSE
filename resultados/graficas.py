"""Figuras de resultados de PointNeXt (A) y PTv3 (B) a partir de runs/*/{log.txt,metrics.json}.

Uso: python resultados/graficas.py   (escribe en resultados/figuras/)
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

AQUI = Path(__file__).parent
RUNS = {"A · PointNeXt": AQUI / "runs/A_PointNeXt", "B · PTv3": AQUI / "runs/B_PTv3"}
OUT = AQUI / "figuras"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
COLOR = {
    "Tabla por frame": "#b5b4ae",
    "Tabla por cluster": "#76756f",
    "A · PointNeXt": "#2a78d6",
    "B · PTv3": "#eb6834",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def leer_log(path):
    pat = re.compile(r"ep\s+(\d+) train ([\d.]+) \| val loss ([\d.]+) cluster ([\d.]+) frame ([\d.]+)")
    ep, frame = [], []
    for linea in path.read_text().splitlines():
        m = pat.match(linea)
        if m:
            ep.append(int(m[1]))
            frame.append(float(m[5]))
    return ep, frame


def curvas(baseline):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre, d in RUNS.items():
        ep, frame = leer_log(d / "log.txt")
        ax.plot(ep, frame, lw=2, color=COLOR[nombre], marker="o", ms=4, label=nombre)
        ax.annotate(nombre, (ep[-1], frame[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.axhline(baseline, color=COLOR["Tabla por cluster"], lw=1.5, ls="--",
               label=f"Tabla por cluster (test) = {baseline:.3f}")
    ax.set_xlabel("época")
    ax.set_ylabel("conteo exacto por frame (validación)")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(-1, 46)
    ax.set_title("Exactitud de conteo por frame durante el entrenamiento", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "curvas_entrenamiento.png", dpi=150)


def barras(metrics):
    base = metrics["A · PointNeXt"]["baseline_test"]
    series = {
        "Tabla por frame": base["frame_tabla_README"],
        "Tabla por cluster": base["frame_suma_clusters"],
        "A · PointNeXt": metrics["A · PointNeXt"]["test"]["frame"],
        "B · PTv3": metrics["B · PTv3"]["test"]["frame"],
    }
    grupos = [("acc", "total"), ("acc_T_0K", "0 K"), ("acc_T_300K", "300 K"),
              ("acc_T_600K", "600 K"), ("acc_forma_mono", "mono"),
              ("acc_forma_conexa", "conexa"), ("acc_forma_dispersa", "dispersa")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.grid(axis="x", visible=False)
    w = 0.2
    for k, (nombre, d) in enumerate(series.items()):
        xs = [i + (k - 1.5) * w for i in range(len(grupos))]
        ys = [d[clave] for clave, _ in grupos]
        ax.bar(xs, ys, width=w - 0.03, color=COLOR[nombre], label=nombre)
    for i, (clave, _) in enumerate(grupos):  # etiqueta solo el grupo "total"
        if clave != "acc":
            continue
        for k, d in enumerate(series.values()):
            ax.text(i + (k - 1.5) * w, d[clave] + 0.01, f"{d[clave]:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.axvline(0.5, color=GRID, lw=1)
    ax.axvline(3.5, color=GRID, lw=1)
    ax.set_xticks(range(len(grupos)), [g for _, g in grupos])
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("conteo exacto por frame (test)")
    ax.set_title("Test por temperatura y por forma del defecto", loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "barras_test.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    metrics = {n: json.loads((d / "metrics.json").read_text()) for n, d in RUNS.items()}
    curvas(metrics["A · PointNeXt"]["baseline_test"]["frame_suma_clusters"]["acc"])
    barras(metrics)
