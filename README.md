# GeneCS

GeneCS is a Python toolkit for synthesizing resource-efficient code surgery procedures for arbitrary CSS quantum stabilizer codes.

The codebase builds an ancilla interaction graph from a chosen logical operator, adds perfect matchings and congestion-aware expander edges, thickens that graph into multiple layers, cellulates the result into cycle checks, and then deforms the original CSS code into a surgery-ready code with bounded qubit and check degree targets.

This repository currently exposes two main workflows:

- `innercode_synthesis(...)`: synthesize a surgery construction for measuring one logical operator inside a single code
- `crosscode_synthesis(...)`: synthesize a merged construction between two codes using an adapter tree

## Features

- Supports arbitrary benchmark CSS codes stored as pickled `qcode` objects
- Builds logical-operator-induced ancilla graphs automatically
- Adds random perfect matchings on overlap sets with stabilizer checks
- Performs congestion-aware graph expansion using Cheeger lower bounds and dynamic cycle-basis maintenance
- Thickens base graphs into layered ancilla systems
- Generates cycle checks and vertex checks for deformed CSS codes
- Exports intermediate graph objects, deformed codes, text logs, and summary CSV files
- Includes scripts for reproducing artifact-style experiments in `docs/artifact`

## Installation

GeneCS is packaged as a small Python project under `src/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Main dependencies from `requirements.txt`:

- `bposd`
- `ldpc`
- `matplotlib`
- `mip`
- `networkx`
- `numpy`
- `pandas`
- `scipy`

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
  Example experiment drivers for overall, merge, scalability, and test runs

docs/distance_verification/
  Distance-check scripts and saved deformation examples

docs/simulation/
  Notebooks, figures, and saved deformation data
```

## Input Format

The synthesis functions expect benchmark files like:

```text
docs/benchmark/BB_72.pkl
docs/benchmark/Surface_21.pkl
```

Each file is loaded with `pickle` and is expected to contain:

```python
data["qcode"]
```

The loaded `qcode` object is expected to provide at least:

- `hx`, `hz`: parity-check matrices
- `lx`, `lz`: logical operators
- `test()`: a validation routine

## Main API

The main implementation lives in [src/genecs/fun.py](/home/junyuzh/HQEC/GeneCS/GeneCS/src/genecs/fun.py).

### 1. Inner-code synthesis

```python
from genecs.fun import innercode_synthesis

innercode_synthesis(
    file_name="BB_72",
    logical_type="X",
    logical_idx=[0],
    qubit_degree=12,
    check_degree=12,
    sample_num=100,
    benchmark_path="./docs/benchmark",
    saving_path="./bin",
)
```

Parameters:

- `file_name`: benchmark code name without `.pkl`
- `logical_type`: `"X"` or `"Z"`
- `logical_idx`: list of logical indices to combine modulo 2
- `qubit_degree`: target maximum qubit participation budget used during graph expansion
- `check_degree`: target degree used during cellulation
- `sample_num`: number of random seeds sampled in congestion-aware expansion
- `benchmark_path`: folder containing benchmark `.pkl` files
- `saving_path`: output folder

What it does:

1. Loads the benchmark code and validates it.
2. Builds a graph whose vertices correspond to qubits in the chosen logical operator.
3. Adds perfect-matching edges using overlaps with opposite-type checks.
4. Runs congestion-aware graph expansion.
5. Thickens the graph into `L` layers.
6. Builds cycle checks by cellulation.
7. Extends the CSS parity checks to produce a deformed code.
8. Saves logs, graph data, code data, and a CSV summary.

### 2. Cross-code synthesis

```python
from genecs.fun import crosscode_synthesis

crosscode_synthesis(
    file_name_1="BB_72",
    file_name_2="BB_72",
    logical_type_1="X",
    logical_type_2="X",
    logical_idx_1=[0, 1],
    logical_idx_2=[0, 1],
    qubit_degree_1=9,
    qubit_degree_2=9,
    check_degree_1=9,
    check_degree_2=9,
    sample_num=100,
    adapter_size=6,
    benchmark_path="./docs/benchmark",
    saving_path="./bin",
)
```

This workflow synthesizes ancilla systems for two codes separately, constructs adapter trees inside each thickened graph, and then merges them into a cross-code surgery construction.

## Command-Line Examples

The repository already includes runnable drivers under `docs/artifact`.

### Overall / single-code experiments

[docs/artifact/overall/prog.py](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/artifact/overall/prog.py)

```bash
cd docs/artifact/overall
python3 prog.py -f BB_72 -lt X -li 0 -s 100 -qd 12 -cd 12
```

### Merge / cross-code experiments

[docs/artifact/merge/prog.py](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/artifact/merge/prog.py)

```bash
cd docs/artifact/merge
python3 prog.py -f1 BB_72 -f2 BB_72 -lt1 X -lt2 X -li1 0 1 -li2 0 1 -qd1 9 -qd2 9 -cd1 9 -cd2 9 -s 100 -a 6
```

### Scalability experiments

[docs/artifact/scalability/prog.py](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/artifact/scalability/prog.py)

```bash
cd docs/artifact/scalability
python3 prog.py -f Surface_21 -lt X -li 0 -s 100 -qd 12 -cd 12
```

### Test artifact script

[docs/artifact/test/prog.py](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/artifact/test/prog.py)

```bash
cd docs/artifact/test
python3 prog.py -f1 BB_72 -f2 BB_72 -lt1 X -lt2 X -li1 0 1 2 -li2 0 1 2 -qd1 9 -qd2 9 -cd1 9 -cd2 9 -s 100 -a 6
```

## Outputs

Both synthesis pipelines create the following subdirectories under the chosen `saving_path`:

- `deform_data/`: text logs and validation output
- `deform_graph/`: pickled graph and synthesis metadata
- `deform_code/`: pickled deformed code objects
- `output.csv`: one-line-per-run summary table

Typical saved files include:

```text
bin/deform_data/BB_72_X_[0].txt
bin/deform_graph/BB_72_X_[0].pkl
bin/deform_code/BB_72_X_[0].pkl
bin/output.csv
```

The CSV currently records:

- file name
- number of new qubits
- number of new checks
- maximum qubit degree
- maximum check degree
- compilation time

## Core Implementation Notes

Important algorithmic pieces are split across the package:

- `create_graph_GeneCS(...)`: maps the chosen logical support to graph vertices
- `construct_perfect_matching_GeneCS(...)`: inserts matching edges induced by stabilizer overlap
- `congestion_aware_expander_sweep_GeneCS(...)`: samples random seeds and selects a low-layer, low-load expander augmentation
- `thicken_graph(...)`: creates the layered ancilla graph
- `cellulation_GeneCS(...)`: derives cycle checks on the thickened graph
- `add_vertex_checks(...)`, `add_perfect_matching_checks(...)`, `append_cycle_basis_checks(...)`: convert graph structure into new CSS parity checks
- `integer_distance_test(...)` and `heuristic_distance_test(...)`: optional exact and heuristic distance checks

## Reproducing Existing Outputs

The repository already contains example outputs under:

- [docs/artifact/test/bin](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/artifact/test/bin)
- [docs/distance_verification](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/distance_verification)
- [docs/simulation](/home/junyuzh/HQEC/GeneCS/GeneCS/docs/simulation)

These are useful as references for expected file formats and naming conventions.

## Limitations and Current Assumptions

- The package currently exports an empty `src/genecs/__init__.py`, so imports are typically done from `genecs.fun`
- `setup.py` does not yet mirror `requirements.txt`, so installing with `pip install -e .` alone is not enough
- Benchmark inputs are expected as pickle files rather than text-based interchange formats
- The README examples assume your benchmark path is `docs/benchmark`; some artifact scripts use relative paths like `../../benchmark` from their own folders
- The code appears specialized to CSS codes and relies on the structure of the `qcode` objects stored in the benchmark files

## License

This project is licensed under the Apache License 2.0. See [LICENSE](/home/junyuzh/HQEC/GeneCS/GeneCS/LICENSE).
