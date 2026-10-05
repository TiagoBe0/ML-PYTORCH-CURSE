# Notebooks orientados al doctorado

Notebooks de Colab que adaptan lo visto en el curso de PyTorch (`../ACurso-PYTORCH/`)
al proyecto de tesis **SPDD**: predecir Δvᵢ (desviación del volumen de Voronoi por átomo)
con una GNN, como descriptor de defectos puntuales en HEA sin referencia estructural.

Fuente de verdad del diseño del proyecto: `~/Documentos/ML-clusters/CONTEXTO.md`.

## Convención

- Nombre: `NN_tema.ipynb` (ej. `01_tensores_posiciones_atomicas.ipynb`).
- Cada notebook arranca con una celda markdown que indica:
  - **Sección del curso** de la que parte (ej. *Section 7 – bucle de entrenamiento*).
  - **Qué cambia** respecto al original y **por qué** sirve para la tesis.
- Datos chicos de prueba en `datos/`; los dumps grandes de LAMMPS quedan fuera
  (en Drive o en el cluster) y se referencian por ruta.

## Índice

| #  | Notebook | Sección del curso | Aplicación a la tesis |
|----|----------|-------------------|-----------------------|
| 01 | [01_tensores_posiciones_atomicas.ipynb](01_tensores_posiciones_atomicas.ipynb) | S4 · La librería PyTorch, Manipulación de tensores | Red FCC/HEA con vacancia → `pos`, `edge_index`, rasgos `x`, etiqueta Δvᵢ (freud). Genera `datos/01_fcc_vacancia.pt` |
| 02 | [02_mlp_baseline_dv.ipynb](02_mlp_baseline_dv.ipynb) | S7–S9 · MLP, bucle de entrenamiento, evaluación, regresión, loss, LR, DataLoader, guardar/cargar | Línea de base: MLP con descriptores G2 (Behler) → Δvᵢ. Split por configuración, ablación, error vs T. Genera `datos/02_mlp_baseline.pt` |
| 03 | [03_hiperparametros_y_entrenamiento.ipynb](03_hiperparametros_y_entrenamiento.ipynb) | S10 · skorch, CV, GridSearch, gestión del entrenamiento, métricas | GroupKFold por configuración, grid vs random search, barrido de r_cut (→ L de la GNN), varianza entre semillas, checkpoints reanudables para SLURM. Necesita `datos/02_dataset.pt`; genera `datos/03_mlp_ajustado.pt` |
| 04 | [04_nube_de_puntos_basico.ipynb](04_nube_de_puntos_basico.ipynb) | S4, S7–S9, S11–S12, S15 · tensores, bucle de entrenamiento, convolución/pooling, data augmentation | Nube de puntos de Ni FCC (`datos/db_1a5_Ni_mini.npz`): PBC, vecindario kNN `[k, 3]`, mini-PointNet (MLP compartido + max), simetrías de Oh, Δvᵢ por átomo vs tabla por coordinación. Figuras del dataset y de una vacancia |
