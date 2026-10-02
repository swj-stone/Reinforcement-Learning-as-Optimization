"""``baseline_results.csv`` 的生成入口。

用法::

    python -m experiment.baselines.cli
    python -m experiment.baselines.cli --out %TEMP%/baseline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

from experiment.baselines.methods import BaselineSolution, greedy_solution, random_solution
from experiment.baselines.seeds import trial_seed
from experiment.data.spec import DATASETS_PROCESSED, E3_NAME, E4_NAME, REPO_ROOT
from experiment.eval.metrics import feasibility_violations
from experiment.eval.reference import item_set_hash

#: 产物放在**仓库根目录并入库**，与 reference_results.csv / data_manifest.json /
#: experiment_config.yaml 一致：datasets/ 整体被 gitignore，把审计凭据放在那里
#: 等于换台机器就无从证明数据未被调换。
DEFAULT_OUT: Path = REPO_ROOT / "baseline_results.csv"

#: 参数从 YAML 读，不硬编码。这正是 `assemble/manifest.py` 的 `build_config`
#: docstring 写明的意图——「让后续阶段读 YAML 而不是回头读 Markdown」：计划若再
#: 修订，只需重新生成 YAML。这两项目前在 Python 侧**没有**可导入的常量，只存在于
#: 转录进 YAML 的 `_EVALUATION` 里，所以读 YAML 也是唯一不必再抄一遍常量的选择。
DEFAULT_CONFIG: Path = REPO_ROOT / "experiment_config.yaml"

#: 列序即产物的一部分。
COLUMNS: tuple[str, ...] = (
    "instance_id",
    "split",
    "method",
    "trial",
    "seed",
    "value",
    "weight_used",
    "violations",
    "item_set_hash",
    "latency_s",
)

#: 语料的子集：只取 test 与两个 stress 集（设计 §1）。
#:
#: 刻意**不含** train / validation：计划 §8 的 E1–E4b 全部落在 test 与 stress 上
#: （E2 是 E1 的分解、不产生新数据，E4b 是 E1 的子集），train/validation 只服务于
#: C 的训练与检查点选择，基线不在其上报告。
SUBSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("test", ("test",)),
    (E3_NAME, ("stress", E3_NAME)),
    (E4_NAME, ("stress", E4_NAME)),
)


def corpus_files(root: Path) -> list[tuple[str, Path]]:
    """三个子集的 ``(标签, 文件)``，顺序固定。

    与 ``experiment/eval/cli.py`` 的同名函数有两处**有意**不同：这里多一层标签
    （本产物有 ``split`` 列，A 的没有），且范围只取三个子集（见 ``SUBSETS``）。
    """
    root = Path(root)
    return [(label, root.joinpath(*parts, "instances.jsonl")) for label, parts in SUBSETS]


def read_instances(files: Sequence[tuple[str, Path]]) -> list[tuple[str, dict]]:
    """展开成 ``(标签, 实例)``，跳过空行。"""
    out: list[tuple[str, dict]] = []
    for label, path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append((label, json.loads(line)))
    return out


def read_config(path: Path) -> tuple[int, int]:
    """从 ``experiment_config.yaml`` 读 ``(evaluation_seed, trials)``。

    Raises:
        KeyError: 缺字段。让它在入口处当场炸掉，好过用一个默认值静默地跑完
            整份语料再产出一份参数不对的产物。
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    return (
        int(evaluation["evaluation_seed"]),
        int(evaluation["random_trials_per_instance"]),
    )


def resolve_out(path: Path, force: bool) -> Path:
    """落盘前的守卫：目标已存在且没给 ``--force`` 就拒绝。

    这不是礼貌。``--out`` 默认指向已入库的 ``baseline_results.csv``，而 ``latency_s``
    是墙钟量——裸跑会把一个已入库文件改写成不同字节、弄脏工作区，而这件事在 A 上
    **真的发生过**。拒绝比事后 ``git checkout`` 便宜。

    Raises:
        FileExistsError: 目标存在且 ``force`` 为假。
    """
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(path)
    return path


def _solution_row(
    label: str,
    method: str,
    instance: dict,
    solution: BaselineSolution,
    trial: int | None,
    seed: int | None,
    latency_s: float,
) -> dict:
    """一行产物。

    ``violations`` 由 ``feasibility_violations`` **算出来**而不是写死 "0"：一个写死的
    常量断言不了任何东西。它是**独立于构造过程**的复核——能抓到"扣容量写错但价值
    算对"这类自洽缺陷（设计 §7）。
    """
    violations = feasibility_violations(instance, solution.selected_indices)
    if violations:
        raise AssertionError(
            f"{instance['instance_id']} 的 {method} 解不可行：{violations} 条违例"
        )
    return {
        "instance_id": instance["instance_id"],
        "split": label,
        "method": method,
        "trial": "" if trial is None else str(trial),
        "seed": "" if seed is None else str(seed),
        "value": str(solution.value),
        "weight_used": str(solution.weight_used),
        "violations": str(violations),
        "item_set_hash": item_set_hash(solution.selected_indices),
        "latency_s": f"{latency_s:.6f}",
    }


def run(
    corpus_root: Path, out_path: Path, *, evaluation_seed: int, trials: int
) -> dict:
    """构造全部基线的解并落盘，返回一份汇总。

    **计时只包住单次构造调用**（设计 §5）：从进入方法到拿到 ``BaselineSolution``
    为止，不含读取实例、不含行构造与写盘。Random 的 10 行各记自己那一次的耗时——
    把十次之和写进每一行是这类产物最常见的错法，它会让 ``latency_s`` 不再是
    "一次构造多贵"，而 D 的 speed 指标正是拿它比三个方法。
    """
    rows: list[dict] = []
    by_method = {"greedy": 0, "random": 0}

    for label, instance in read_instances(corpus_files(corpus_root)):
        start = time.perf_counter()
        greedy = greedy_solution(instance)
        greedy_latency = time.perf_counter() - start
        rows.append(_solution_row(label, "greedy", instance, greedy, None, None, greedy_latency))
        by_method["greedy"] += 1

        for trial in range(trials):
            seed = trial_seed(instance["instance_id"], trial, evaluation_seed)
            start = time.perf_counter()
            solution = random_solution(instance, seed)
            latency = time.perf_counter() - start
            rows.append(_solution_row(label, "random", instance, solution, trial, seed, latency))
            by_method["random"] += 1

    # 行序是产物的一部分。``trial`` 按**数值**排：按字符串排会把 "10" 排到 "2"
    # 前面（"0" < "1" < "10" < "2"）——10 次试验时试验号恰好全是单字符，看不出
    # 来，所以这个坑由一个 12 次试验的测试钉住。贪心没有试验号，用 -1 让它排在该
    # 实例的随机行之前。
    rows.sort(key=lambda row: (row["instance_id"], row["method"], int(row["trial"] or -1)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 关掉 Python 的换行翻译，lineterminator="\n" 固定为 LF。
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {"by_method": by_method, "out": out_path, "rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为冻结语料构造 Random 与 Greedy 两个基线"
    )
    parser.add_argument(
        "--corpus", type=Path, default=DATASETS_PROCESSED, help="冻结语料根目录"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="experiment_config.yaml 的位置"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="baseline_results.csv 的位置"
    )
    parser.add_argument("--force", action="store_true", help="目标已存在时也覆写")
    return parser


def main() -> int:
    # 与 generate/collect/assemble/eval 的入口一致：重定向到文件时 Windows 用 ANSI
    # 代码页编码 stdout，中文标签会抛 UnicodeEncodeError。
    #
    # stderr 一并重设：本函数是**本项目第一个往 stderr 写字**的入口（覆写守卫那两行），
    # 而重设之前它们在 GBK 控制台上是乱码——一条告诉用户「加 --force」的提示，读不出来
    # 就等于没有。探针实测过修复前后的两种输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    try:
        out_path = resolve_out(args.out, args.force)
    except FileExistsError:
        print(f"产物已存在，未覆写: {args.out}", file=sys.stderr)
        print("确要覆写请加 --force。", file=sys.stderr)
        return 1

    evaluation_seed, trials = read_config(args.config)
    start = time.perf_counter()
    summary = run(args.corpus, out_path, evaluation_seed=evaluation_seed, trials=trials)

    # 刻意**不**打印 A 的入口里那行"平凡解: N"：那里数的是 value == 0 的行，在 A
    # 是对的；在这里 Random 抽到全反面时会返回空集、value 为 0，那是合法的一次抽样，
    # 不是计划 §5 说的"没有物品装得下的平凡零最优实例"。
    print(f"语料根目录: {args.corpus}")
    print(f"产物:       {summary['out']}")
    print(f"实例行数:   {summary['rows']}")
    print(f"  greedy:   {summary['by_method']['greedy']}")
    print(f"  random:   {summary['by_method']['random']}")
    print(f"试验次数:   {trials}（来自 {args.config}）")
    print(f"墙钟总耗时: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
