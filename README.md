# PSO-GA-VRU-Sensing

MATLAB implementation of a PSO-based energy-efficient VRU sensing framework with safety and responsiveness constraints, including robust multi-run PSO–GA comparison (mean ± std) for stability analysis.

## Overview
This repository contains MATLAB scripts implementing a particle swarm optimization (PSO) framework for energy-efficient sensing of vulnerable road users (VRUs). The framework jointly optimizes sensor activation and duty cycling while enforcing energy, safety, and responsiveness constraints. A genetic algorithm (GA) baseline is included for comparative evaluation.

The implementation supports:
- Binary sensor activation decisions
- Continuous duty-cycle optimization
- Constraint handling via penalty-based fitness design
- Single-run performance diagnostics and multi-run statistical analysis

---

## Note on Responsiveness Constraint
This implementation explicitly addresses the zero-responsiveness issue by incorporating a minimum responsiveness constraint directly into the fitness function. Responsiveness is computed only over active sensors and enforced via a penalty term, ensuring physically meaningful and non-degenerate solutions.

---

## Generated Metrics and Plots
The provided scripts generate the following performance metrics and visualizations:

- Energy consumption over iterations
- Safety performance over iterations
- Responsiveness evolution over iterations
- Penalty terms for:
  - Energy constraint violation
  - Safety constraint violation
  - Responsiveness constraint violation
- Fitness convergence curves
- Mean ± standard deviation comparison of PSO vs. GA over multiple independent runs

These plots support convergence analysis, constraint satisfaction verification, and algorithm stability assessment.
