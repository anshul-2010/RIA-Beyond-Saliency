import argparse
import json

import numpy as np
import pytest

from ria.runlog import Run, load_config, read_results, set_seed


def test_run_writes_config_and_results(tmp_path):
    cfg = {"seed": 3, "model": "m", "results_dir": tmp_path}
    with Run("unit", cfg) as run:
        run.log({"i": 0, "G3": np.float32(0.5), "players": frozenset({"b", "a"})})
        run.log({"i": 1, "G3": -0.25})
    meta = json.loads((run.dir / "config.json").read_text(encoding="utf-8"))
    assert meta["args"]["model"] == "m" and "git_commit" in meta["env"]
    rows = read_results(run.dir)
    assert rows == [{"i": 0, "G3": 0.5, "players": ["a", "b"]}, {"i": 1, "G3": -0.25}]


def test_runs_with_same_stamp_get_distinct_dirs(tmp_path):
    a = Run("same", {"results_dir": tmp_path})
    b = Run("same", {"results_dir": tmp_path})
    a.close(); b.close()
    assert a.dir != b.dir


def test_seed_is_reproducible():
    set_seed(7); x = np.random.rand(3)
    set_seed(7); y = np.random.rand(3)
    assert np.array_equal(x, y)


def test_config_file_defaults_and_cli_override(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"operator": "blur", "seed": 5}), encoding="utf-8")
    p = argparse.ArgumentParser(); p.add_argument("--operator", default="drop")
    cfg = load_config(p, ["--config", str(f)])
    assert cfg["operator"] == "blur" and cfg["seed"] == 5
    p = argparse.ArgumentParser(); p.add_argument("--operator", default="drop")
    assert load_config(p, ["--config", str(f), "--operator", "mean"])["operator"] == "mean"


def test_config_file_rejects_unknown_keys(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"opertor": "blur"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(argparse.ArgumentParser(), ["--config", str(f)])


def test_run_dir_resumes_and_lists_done_keys(tmp_path):
    d = tmp_path / "fixed"
    with Run("x", {"results_dir": tmp_path}, run_dir=d) as run:
        run.log({"key": "a"})
    with open(d / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"key": "b"')                     # a line cut off mid-write
    with Run("x", {"results_dir": tmp_path}, run_dir=d) as run:
        assert run.done_keys() == {"a"}
        run.log({"key": "c"})
    assert (d / "config.json").exists() and len(list(d.glob("config_resume_*.json"))) == 1
