# GeneCS

GeneCS is a Python toolkit for synthesizing resource-efficient code surgery procedures for quantum stabilizer codes.

The codebase builds an ancilla interaction graph from a chosen logical operator, adds perfect matchings and congestion-aware expander edges, thickens that graph into multiple layers, cellulates the cycle checks, and then deforms the original code into a surgery-ready code with bounded qubit and check degree targets.

## Workflows

This repository currently exposes two main workflows:

- `innercode_synthesis(...)`: synthesize a surgery construction for measuring logical operator2 inside a single code
- `crosscode_synthesis(...)`: synthesize a merged surgery construction between two codes using an adapter 


## Installation

GeneCS is packaged as a Python project under `src/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Repository Layout

```text
src/genecs/
  fun.py       Main synthesis pipeline and graph algorithms
  deform.py    Code deformation, parity-check extension, and distance tests
  adapter.py   Adapter and merge helpers
  utils.py     Graph utilities, Cheeger bound, visualization, and checks

docs/benchmark/
  Pickled benchmark codes used as synthesis inputs

docs/artifact/
  Example experiment drivers for overall, cross-code communication, scalability studies report in the paper

docs/distance_verification/
  Distance-check scripts, including integer programming strategy and heuristic strategy

docs/simulation/
  Notebook for logical error rate simulation, require additional software to run
```

## Future work

- Add functionality to produce surgery constructions for non-CSS stabilizer codes
- Connect multiple code patches within a unified synthesis workflow
