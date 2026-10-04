# Industry-LLM-Timeseires

[中文](README.zh-CN.md) | English

An industrial time-series AI learning repository focused on reproducing classic papers, covering tasks such as anomaly detection, forecasting, classification, and time-series representation (including the LLM direction).

> Entry point for AI agents / maintainers: [AGENTS.md](AGENTS.md) (architecture rules, operating procedures, acceptance checklist).

## Design principles

1. **Single environment, directory-level decoupling**: the whole repository shares one uv environment (root `.venv`); all dependencies live in the root `pyproject.toml`. Each algorithm is a self-contained source directory — not packaged, no cross-imports; deleting any algorithm directory does not affect the others.
2. **Rule of three**: shared code is extracted only after the same logic appears 3 times; a little duplication is preferred to keep algorithms fully decoupled.
3. **Paper ↔ algorithm ↔ code three-way mapping**: every paper has a close-reading note (`docs/papers/`); every algorithm README links back to its paper note; code docstrings cite paper sections/equation numbers; the master table is `docs/README.md`.
4. **Data/code separation with standardized formats**: datasets live in `data/` and are registered (`data/README.md`); data files are not committed to Git. Preprocessed artifacts follow a uniform format (`train.csv` / `test.csv` / `meta.json`); algorithms load data via `load_dataset(name)` — **switching datasets changes parameters, not code**. Results go to `outputs/<algo>/<dataset>/`; besides `metrics.json` (with a parameter snapshot), iterative algorithms must produce `history.csv` + `curves.png`, and non-iterative algorithms must produce at least one result visualization — intuitive for humans beyond the numbers. `scripts/collect_results.py` automatically aggregates the algorithm × dataset comparison table into `docs/benchmarks.md`.

## Directory structure

```
├── algorithms/             # All algorithms, grouped by task
│   └── anomaly_detection/
│       └── isolation_forest/   # Example algorithm (also a living template instance)
├── templates/algorithm-template/  # Template for new algorithms
├── scripts/                # new_algorithm.py (scaffolding), test_all.py (full tests),
│                           # collect_results.py (result aggregation), preprocess_<dataset>.py (preprocessing)
├── data/                   # Shared datasets (raw/ processed/; contents not committed to Git)
│   └── README.md           # Dataset registry
├── docs/
│   ├── README.md           # Paper↔algorithm mapping master table
│   ├── papers/             # Paper close-reading notes
│   └── benchmarks.md       # Multi-algorithm comparison table per dataset
├── outputs/                # Run artifacts (not committed to Git)
└── pyproject.toml          # All dependencies + ruff configuration
```

Reserved task categories (created when needed): `forecasting`, `classification`, `representation` (representation/LLM direction).

## Quick start

```bash
uv sync                                          # Install all dependencies (single environment)

# Run the example algorithm (default synthetic dataset; --dataset switches among registered datasets)
cd algorithms/anomaly_detection/isolation_forest
uv run python train.py                           # Results written to outputs/isolation_forest/<dataset>/
uv run pytest -q                                 # Single-algorithm tests

# Back to the repository root
cd ../../..
uv run python scripts/test_all.py                # All algorithm tests (subprocess per directory)
uv run python scripts/collect_results.py         # Aggregate results -> docs/benchmarks.md
uv run ruff check .                              # Lint
```

## Adding a new algorithm

```bash
uv run scripts/new_algorithm.py <category> <algorithm_name>
# e.g.: uv run scripts/new_algorithm.py forecasting dlinear
```

Then (see [CONTRIBUTING.md](CONTRIBUTING.md) for details):

1. Fill in the metadata of the new directory's `README.md` (paper-note link, datasets, status)
2. Create a paper note in `docs/papers/` and register a row in the `docs/README.md` master table
3. Implement `data.py` / `model.py` / `train.py`, then `cd` into the directory to run and verify
4. If new dependencies are needed: `uv add <package>` in the root directory

## Maintenance rules at a glance

- Naming: directories and files always `snake_case`; paper notes `<algorithm>_<venue_year>.md`
- Tests: `uv run pytest` inside the algorithm directory; full suite via `scripts/test_all.py`
- Data: put raw data in `data/raw/` and register it in `data/README.md`; preprocessing scripts go in `scripts/preprocess_<dataset>.py`
- Results: after runs, execute `uv run python scripts/collect_results.py` to refresh the comparison table (never edit `docs/benchmarks.md` by hand)
- Dependencies: only `uv add` (`uv pip install` is forbidden — it is wiped by `uv sync`); the CUDA build of PyTorch is pinned in `pyproject.toml`
- Before pushing to GitHub: go through the privacy checklist in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

[MIT](LICENSE)
