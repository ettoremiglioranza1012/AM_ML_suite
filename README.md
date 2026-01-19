# AM - Additive Manufacturing Topology Optimization

**Sistema ibrido HPC + AI per Topology Optimization in Additive Manufacturing**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Vision

Costruire un sistema end-to-end per la **Topology Optimization** di componenti aeronautici in Metal AM, combinando:

1. **Solver Numerico Python** → Ground Truth per validazione e prototipazione
2. **Motore HPC C++/MPI** → Generazione massiva di dataset di training
3. **Modello AI (Deep Learning)** → Inferenza rapida (ms vs minuti)

---

## 🏗️ Architettura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AM TOPOLOGY OPTIMIZATION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   CORE      │    │  NUMERICAL  │    │     AI      │                  │
│  │             │    │   (Python)  │    │  (PyTorch)  │                  │
│  │ • geometry  │◄───┤             │    │             │                  │
│  │ • loads     │    │ • fem.py    │    │ • model.py  │                  │
│  │             │◄───┤ • topopt.py │    │ • inference │                  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
│         │                  │                  │                          │
│         │      Shared      │   Ground Truth   │   Fast Inference        │
│         │      Definitions │   Validation     │   (Trained Model)       │
│         │                  │                  │                          │
├─────────┴──────────────────┴──────────────────┴─────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     C++ ENGINE (HPC Data Factory)                 │   │
│  │                                                                    │   │
│  │   • High-performance FEM solver                                   │   │
│  │   • MPI parallelization for massive dataset generation            │   │
│  │   • Produces training data for AI model                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flusso Dati

```
                    ┌───────────────────┐
                    │  Problem Definition│
                    │  (geometry, loads) │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │  Numerical │   │  C++ HPC   │   │    AI      │
     │   Solver   │   │  Engine    │   │  Inference │
     │  (Python)  │   │  (Future)  │   │  (Future)  │
     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
           │                │                │
           │     ~minutes   │   ~seconds     │   ~milliseconds
           │                │   (per case)   │
           ▼                ▼                ▼
     ┌────────────────────────────────────────────┐
     │              Density Field                 │
     │           (3D voxel array)                 │
     └────────────────────────────────────────────┘
```

---

## 📁 Struttura Progetto

```
AM/
│
├── main.py                      # 🎮 Entry point unificato
│                                #    --mode numerical | ai
│
├── src/
│   └── am/                      # 📦 Package Python principale
│       │
│       ├── core/                # 🔧 Definizioni condivise
│       │   ├── geometry.py      #    Dominio voxel, design space
│       │   └── loads.py         #    Load cases, boundary conditions
│       │
│       ├── numerical/           # 🧮 Solver Python (Ground Truth)
│       │   ├── fem.py           #    Assemblaggio matrice K, solver
│       │   ├── topopt.py        #    Loop SIMP, Optimality Criteria
│       │   └── README.md        #    Documentazione dettagliata
│       │
│       └── ai/                  # 🤖 Modulo Deep Learning
│           ├── model.py         #    Architettura 3D U-Net
│           └── inference.py     #    Pipeline di inferenza
│
├── cpp_engine/                  # ⚡ HPC Data Factory (C++/MPI)
│   ├── CMakeLists.txt           #    Build configuration
│   └── src/                     #    Sorgenti C++ (future)
│
├── data/
│   └── brk_a_01/                # 📊 Dati caso pilota
│       ├── density_field.npy    #    Campo densità output
│       └── metadata.json        #    Metadati run
│
├── notebooks/                   # 📓 Jupyter notebooks
│   ├── 01_brk_a_01_topopt.ipynb #    Ottimizzazione interattiva
│   └── 02_visualize_results.ipynb #  Visualizzazione 3D
│
├── pyproject.toml               # ⚙️ Configurazione progetto
└── uv.lock                      # 🔒 Lock dipendenze
```

---

## 🧩 Moduli

### `am.core` - Definizioni Condivise

Contiene le classi che definiscono il problema fisico, usate trasversalmente da tutti i solver:

| Classe | Descrizione |
|--------|-------------|
| `VoxelDomain` | Griglia 3D con marking design/non-design/void |
| `LoadCase` | Caso di carico con forze e vincoli |
| `BoundaryCondition` | Condizioni al contorno (DOF vincolati) |
| `PointLoad` | Carichi puntuali applicati |

```python
from am.core.geometry import create_bracket_domain, VoxelDomain
from am.core.loads import create_brk_a_01_static_case_1, LoadCase
```

### `am.numerical` - Solver Python

Il prototipo originale, ora incapsulato come modulo di validazione:

| Modulo | Responsabilità |
|--------|----------------|
| `fem.py` | Matrice di rigidezza K (esaedro 8 nodi), solver lineare |
| `topopt.py` | Loop SIMP: update densità, filtro, Optimality Criteria |

```python
from am.numerical.fem import MaterialProperties
from am.numerical.topopt import SIMPOptimizer, SIMPParams
```

### `am.ai` - Deep Learning (In Sviluppo)

Modulo per inferenza rapida con reti neurali:

| Modulo | Responsabilità |
|--------|----------------|
| `model.py` | Architettura 3D U-Net per predizione densità |
| `inference.py` | Pipeline di inferenza con modello pre-addestrato |

```python
# Future usage
from am.ai.inference import AIOptimizer
optimizer = AIOptimizer(model_path="models/topopt_unet.pt")
density = optimizer.predict(domain, load_cases)
```

### `cpp_engine` - HPC Data Factory (Planned)

Solver C++ ad alte prestazioni per generazione massiva di dataset:

- **Linguaggio:** C++20
- **Dipendenze:** Eigen, OpenMP, MPI (opzionale)
- **Scopo:** Generare migliaia di esempi (input, density_field) per training AI

---

## 🚀 Quick Start

### Installazione

```bash
# Clone repository
git clone <repo-url>
cd AM

# Setup ambiente (con uv)
uv sync

# Oppure con pip
pip install -e .
```

### Esecuzione

```bash
# Solver numerico (default)
uv run python main.py --mode numerical --resolution 2.0 --max-iter 50

# Con parametri personalizzati
uv run python main.py -m numerical -r 1.0 --volume-fraction 0.30 -o data/custom

# AI inference (richiede modello addestrato)
uv run python main.py --mode ai --model-path models/topopt_unet.pt
```

### Opzioni CLI

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `--mode, -m` | `numerical` | Modalità: `numerical` o `ai` |
| `--resolution, -r` | `1.0` | Risoluzione voxel [mm] |
| `--volume-fraction, -vf` | `0.25` | Frazione di volume target |
| `--max-iter` | `50` | Iterazioni massime (numerical) |
| `--output-dir, -o` | `data/brk_a_01` | Directory output |
| `--model-path` | - | Path modello AI (richiesto per `--mode ai`) |

---

## 🔬 Caso Pilota: BRK-A-01

**Staffa aeronautica** per supporto attuatore (Pylon/Engine Bracket)

| Parametro | Valore |
|-----------|--------|
| **Materiale** | Ti6Al4V (E=113.8 GPa, ν=0.342) |
| **Processo** | L-PBF (Metal AM) |
| **Dominio** | 120 × 60 × 80 mm |
| **Risoluzione** | 1 mm (576,000 voxel) |
| **Carico** | 15 kN verticale su occhiello |
| **Volume target** | 25% (rimozione 75% materiale) |

### Output

- `density_field.npy` - Campo densità 3D [0, 1]
- `metadata.json` - Metadati run (compliance, iterazioni, tempo)

---

## 📊 Roadmap

### ✅ v0.1 - Prototipo Python
- [x] Dominio voxel con NDS
- [x] Assemblaggio matrice K 3D
- [x] Solver FEM sparse
- [x] Loop SIMP con OC
- [x] Filtro densità
- [x] Visualizzazione 3D

### 🚧 v0.2 - Refactoring Architettura (Corrente)
- [x] Struttura a pacchetto (`src/am/`)
- [x] Separazione core/numerical/ai
- [x] Entry point unificato con CLI
- [x] Placeholder modulo AI
- [x] Placeholder C++ engine

### 📋 v0.3 - AI Module
- [ ] Implementazione 3D U-Net (PyTorch)
- [ ] Data loader per coppie (input, density)
- [ ] Training script
- [ ] Metriche validazione (IoU, compliance error)

### 📋 v0.4 - C++ HPC Engine
- [ ] FEM solver in C++/Eigen
- [ ] Parallelizzazione OpenMP
- [ ] Generazione dataset massivo
- [ ] Export binario per Python

### 📋 v1.0 - Sistema Completo
- [ ] Modello AI addestrato
- [ ] Validazione cross-solver
- [ ] Deployment inference
- [ ] Documentazione completa

---

## 📚 Riferimenti

1. Bendsøe, M.P., Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*
2. Andreassen, E. et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code*
3. Nie, Z. et al. (2021). *TopologyGAN: Topology Optimization Using GANs*
4. Sosnovik, I., Oseledets, I. (2019). *Neural Networks for Topology Optimization*

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*Last updated: January 19, 2026*
