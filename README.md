# PSO-GA-VRU-Sensing
MATLAB implementation of a PSO-based energy-efficient VRU sensing framework with safety and responsiveness constraints, including robust multi-run PSO–GA comparison (mean ± std) for stability analysis.

## Overview
This repository contains MATLAB scripts implementing a particle swarm optimization (PSO) framework for energy-efficient sensing of vulnerable road users (VRUs). The framework jointly optimizes sensor activation and duty cycling while enforcing energy, safety, and responsiveness constraints. A genetic algorithm (GA) baseline is included for comparative evaluation.

---

## Note on Responsiveness Constraint
This implementation explicitly addresses the zero-responsiveness issue by incorporating a minimum responsiveness constraint directly into the fitness function. Responsiveness is computed only over active sensors and enforced via a penalty term, ensuring physically meaningful and non-degenerate solutions.
