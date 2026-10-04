# CONTRIBUTING

[中文](CONTRIBUTING.zh-CN.md) | English

This is a personal learning repository; the conventions below ensure long-term **maintainability, reproducibility, and privacy safety**.

## 1. Adding a new algorithm

```bash
uv run scripts/new_algorithm.py <category> <algo_name>
```

The scaffold copies the skeleton from `templates/algorithm-template/` and replaces placeholders. Then complete, in order:

1. **Algorithm README**: fill in the metadata table — relative link to the paper note, paper source, reproduction status (`not started / in progress / reproduced`), registered dataset name, result summary.
2. **Paper note**: create `<algo>_<venue_year>.md` under `docs/papers/` (template in `docs/README.md`), covering motivation / method / experimental setup / reproduction notes.
3. **Register in the master table**: add a row to the mapping table in `docs/README.md`: paper | paper note | algorithm implementation | dataset | status.
4. **Implement the code**: `data.py` (`load_dataset` reads the standard format), `model.py`, `train.py` (argparse entry; metrics written to `outputs/<algo>/<dataset>/metrics.json`).
5. **Code ↔ paper annotation**: docstrings of key classes/functions cite the corresponding paper section or equation number, e.g. `Implements Eq. (3) of <paper>`.
6. **Verify**: `uv run pytest -q` passes inside the algorithm directory; `uv run ruff check .` reports zero warnings in the root; `uv run python scripts/collect_results.py` refreshes the comparison table.

### Adding dependencies

Run `uv add <package>` in the root (the whole repository shares a single environment). Do not create a separate environment for an individual algorithm.

**`uv pip install` is forbidden**: it writes to the environment but not to `pyproject.toml`, and `uv sync` strictly realigns the environment to the declared dependencies and **removes** it. All dependencies must land in `pyproject.toml`.

**The CUDA build of PyTorch is pinned**: via `[tool.uv.sources]` + `[[tool.uv.index]]` pointing at the official `cu130` index (for RTX 50-series); to upgrade, change the version number — do not override with `uv pip install`.

**Dependency-conflict fallback** (expected to be rare): if an algorithm genuinely needs a version conflicting with others, `uv init` an independent environment only inside that algorithm's directory and mark it prominently in its README; the rest of the repository stays unchanged.

## 2. Data and comparison conventions

- Data goes in `data/raw/<dataset>/`; preprocessed artifacts go in `data/processed/<dataset>/`; **data files are not committed to Git**.
- Preprocessed artifacts must follow the **standard format** (`train.csv` / `test.csv` / `meta.json`; fields in `data/README.md`); preprocessing scripts live in `scripts/preprocess_<dataset>.py`, one script per dataset.
- Every dataset must be registered in `data/README.md`: source, license, format fields, preprocessing and split method — the prerequisite for comparable multi-algorithm results.
- Algorithms load data via `load_dataset(name)`; `train.py` must support CLI arguments like `--dataset`/`--seed`; **switching datasets changes parameters, not code**.
- Each run writes `outputs/<algo>/<dataset>/`: `metrics.json` (with parameter snapshot) is the machine-readable layer, from which `scripts/collect_results.py` automatically refreshes `docs/benchmarks.md` (**manual edits forbidden**); iterative algorithms must also write `history.csv` and render `curves.png`, and non-iterative algorithms must output at least one result visualization — so a human looking at the output understands the training/result process, not just a pile of numbers.

## 3. Shared-code discipline (rule of three)

Do not extract a shared module until the same logic has been duplicated across algorithms **3 times**. When extraction is truly needed, first record the three call sites in an Issue/note, then consider creating `libs/` (update this file and the README at that point).

## 4. Pre-commit privacy checklist

Confirm each item before pushing to GitHub:

- [ ] No local absolute paths (e.g. `C:\Users\<username>`, user home directories)
- [ ] No real names, personal emails, phone numbers, or other personal information (author attribution excepted)
- [ ] No API Keys / Tokens / passwords (check prefixes like `sk-`, `AKIA`, `ghp_`; `.env` is gitignored)
- [ ] No data files or model weights (`data/raw/`, `data/processed/`, `outputs/`, `*.pt`, etc. are gitignored)
- [ ] No paper PDFs (`*.pdf` is gitignored; notes contain links only)
- [ ] Every entry in the `git status` pending list reviewed individually

## 5. Code style

- Checked uniformly by ruff (`uv run ruff check .`), line width 88, automatic import-sort fix: `uv run ruff check --fix .`
- Comments and documentation in Chinese; code identifiers in English
