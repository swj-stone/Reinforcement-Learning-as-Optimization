"""命令行入口的参数与退出码。"""

import json

import pytest

from experiment.data.assemble import cli

CELL = "20x0.25"
BASE_ARGV = ["--cell", CELL, "--synthetic-quota", "1", "--real-quota", "4", "--no-stress"]


def _argv(tmp_path, lots_csv, fixture_corpus) -> list[str]:
    return [
        "cli",
        "--out", str(tmp_path / "processed"),
        "--lots", str(lots_csv),
        "--generation", str(fixture_corpus),
        "--manifest", str(tmp_path / "data_manifest.json"),
        "--config", str(tmp_path / "experiment_config.yaml"),
        *BASE_ARGV,
    ]


def test_main_writes_every_artifact_and_exits_zero(
    tmp_path, lots_csv, fixture_corpus, monkeypatch
):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    assert cli.main() == 0
    assert (tmp_path / "processed" / "train" / "instances.jsonl").is_file()
    assert (tmp_path / "processed" / "validation" / "instances.jsonl").is_file()
    assert (tmp_path / "processed" / "test" / "instances.jsonl").is_file()
    doc = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["counts"]["train"] == {"synthetic": 4, "real": 4, "total": 8}


def test_main_reports_the_counts_it_wrote(
    tmp_path, lots_csv, fixture_corpus, monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    cli.main()
    printed = capsys.readouterr().out
    assert "train" in printed
    assert "50:50" in printed
    # 每个核心 split 都是 4 合成 + 4 真实
    assert printed.count("合成 4 + 真实 4 = 8") == 3


def test_main_skips_stress_when_asked(tmp_path, lots_csv, fixture_corpus, monkeypatch):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    cli.main()
    assert not (tmp_path / "processed" / "stress").exists()


def test_parse_cell_accepts_the_plan_spelling():
    assert cli.parse_cell("20x0.25") == (20, 0.25)
    assert cli.parse_cell("200x0.75") == (200, 0.75)


def test_parse_cell_rejects_garbage():
    with pytest.raises(Exception):
        cli.parse_cell("twenty")
    with pytest.raises(Exception):
        cli.parse_cell("20")


def test_default_paths_come_from_spec():
    from experiment.data import spec

    args = cli.build_parser().parse_args([])
    assert args.out == spec.DATASETS_PROCESSED
    assert args.manifest == spec.DATA_MANIFEST_PATH
    assert args.config == spec.EXPERIMENT_CONFIG_PATH
    assert args.lots == spec.REAL_DIR / "lots_clean.csv"
    assert args.generation == spec.GENERATION_DIR
    assert args.cell is None
    assert args.no_stress is False
