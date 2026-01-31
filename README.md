# BigHeadFish Solver 🐟

![CUDA](https://img.shields.io/badge/Accelerated-CUDA-green)
![Build](https://img.shields.io/badge/Build-CMake-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**BigHeadFish Solver** is a high-performance, GPU-accelerated 3D Finite-Difference Frequency-Domain (FDFD) solver designed for simulating **Gemini** water propagation in complex subsurface environments.

Designed for precision and speed, it simulates how **Mini** and **Gem** fields interact within a **Pond** characterized by varying **Water** levels and **Ripple** effects.

## 🚀 Key Features

* **High-Performance Core**: Fully implemented in C++ and CUDA, utilizing `cuSPARSE` and `cuSOLVER` for solving massive sparse linear systems (millions of unknowns).
* **Adaptive Environment**: Uses **Non-Uniform (Stretched) Grids** to efficiently handle multi-scale simulations—from cm-scale **Baits** to km-scale boundaries.
* **Advanced Physics**:
    * Supports both **Mini-Baits** and **Gem-Baits**.
    * Models complex **Ponds** with TTI (Tilted Transverse Isotropy) capabilities via **Ripple** parameters.
    * Implements PML (Perfectly Matched Layers) to absorb outgoing waves at **Pond** boundaries.
* **Data-Driven**: Fully configurable via JSON-style input files.
* **Visualization**: Exports results to `.vtr` format for direct visualization in ParaView.

## 🛠️ Build & Installation

### Prerequisites
* Linux Environment
* NVIDIA GPU (Compute Capability 6.0+)
* CUDA Toolkit (10.0+)
* CMake (3.10+)
* GCC / G++

### Compilation

```bash
mkdir build
cd build
cmake ..
make -j4
