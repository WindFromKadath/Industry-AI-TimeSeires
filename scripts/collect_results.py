"""Aggregate outputs/<algo>/<dataset>/metrics.json into docs/benchmarks.md.

Run from repo root:
    uv run python scripts/collect_results.py

docs/benchmarks.md is fully overwritten — never edit it by hand.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TARGET = REPO_ROOT / "docs" / "benchmarks.md"

HEADER = (
    "# 基准对比\n\n"
    "<!-- 本文件由 scripts/collect_results.py 自动生成,请勿手改 -->\n\n"
    "同一数据集上多个算法的结果对比,指标取自 "
    "`outputs/<algo>/<dataset>/metrics.json`。\n"
)


def main() -> int:
    by_dataset: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for path in sorted(OUTPUTS_DIR.glob("*/*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        by_dataset[path.parent.name].append((path.parent.parent.name, record))

    lines = [HEADER]
    if not by_dataset:
        lines.append("\n(暂无运行结果)\n")
    for dataset in sorted(by_dataset):
        entries = by_dataset[dataset]
        metric_names = sorted({m for _, r in entries for m in r.get("metrics", {})})
        lines.append(f"\n## {dataset}\n\n")
        lines.append("| 算法 | " + " | ".join(metric_names) + " | 参数 | 运行日期 |\n")
        lines.append("|---" * (len(metric_names) + 3) + "|\n")
        for algo, record in entries:
            metrics = record.get("metrics", {})
            cells = [str(metrics.get(m, "-")) for m in metric_names]
            params = {
                k: v for k, v in record.get("params", {}).items() if k != "dataset"
            }
            params_str = "`" + json.dumps(params, ensure_ascii=False) + "`"
            date = str(record.get("timestamp", ""))[:10]
            row = (
                f"| {algo} | " + " | ".join(cells) + f" | {params_str} | {date} |\n"
            )
            lines.append(row)

    TARGET.write_text("".join(lines), encoding="utf-8")
    n_runs = sum(len(v) for v in by_dataset.values())
    print(f"wrote {TARGET.relative_to(REPO_ROOT)} ({n_runs} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
