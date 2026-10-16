"""
Sim-free tests for damagesim/llm_params/ground_truth.py (Notion task D0.3).

Run from the repo root in any env with pytest, for example:
    uv run --no-project --python 3.10 --with pytest --with numpy --with pyyaml \
        python -m pytest scripts/tests/test_llm_params_ground_truth.py -q
"""

import ast
import csv
import json
import os
import subprocess
import sys

import pytest

from damagesim.llm_params import ground_truth as gt


def _row(backend, key):
    rows = [r for r in gt.export(backend) if r["entry_key"] == key]
    assert len(rows) == 1, (backend, key)
    return rows[0]


# ── Table reading and row export ────────────────────────────────────────

def test_row_counts():
    rs, og = gt.export("robocasa"), gt.export("omnigibson")
    assert len(rs) == 37 and sum(r["is_target"] for r in rs) == 33
    assert len(og) == 47 and sum(r["is_target"] for r in og) == 45


def test_kinds():
    rs = {r["entry_key"]: r["kind"] for r in gt.export("robocasa")}
    assert rs["default"] == rs["fixture_default"] == "default"
    assert rs["Panda"] == rs["PandaOmron"] == "robot"
    assert rs["stool"] == "fixture" and rs["egg"] == "object"
    og = {r["entry_key"]: r["kind"] for r in gt.export("omnigibson")}
    assert og["agent"] == "robot" and og["default"] == "default" and og["laptop"] == "category"


def test_spot_values():
    egg = _row("robocasa", "egg")
    assert (egg["damage_threshold"], egg["damage_scale"],
            egg["impact_damage_sensitivity"], egg["qs_damage_sensitivity"]) == (15.0, 1.0, 10.0, 1.0)
    lap = _row("omnigibson", "laptop")
    assert (lap["electrical_threshold"], lap["electrical_scale"], lap["electrical_water_system"]) == (20.0, 5.0, "water")
    assert json.loads(lap["electrical_link_thresholds"]) == {
        "keyboard": {"damage_threshold": 50.0, "scale": 8.0},
        "screen": {"damage_threshold": 50.0, "scale": 10.0},
    }
    agent = _row("omnigibson", "agent")
    assert (agent["heating_threshold"], agent["cooling_threshold"], agent["thermal_scale"]) == (45.0, -20.0, 0.1)
    assert "robot_overrides" in agent["notes"]


def test_thermal_electrical_presence():
    for b in ("robocasa", "omnigibson"):
        for r in gt.export(b):
            is_og = b == "omnigibson"
            assert (r["heating_threshold"] is not None) == (is_og and r["entry_key"] == "agent")
            assert (r["electrical_threshold"] is not None) == (
                is_og and r["entry_key"] in {"agent", "laptop", "book", "comic_book"})


def test_link_override_flags():
    assert _row("robocasa", "microwave")["has_link_overrides"] is True
    og_mw = _row("omnigibson", "microwave")
    assert og_mw["has_link_overrides"] is False and "misnested" in og_mw["notes"]
    assert og_mw["has_damageable_links"] is True


def test_source_lines_point_at_keys():
    for b in ("robocasa", "omnigibson"):
        for r in gt.export(b):
            line = (gt.REPO_ROOT / r["source_file"]).read_text().splitlines()[r["source_line"] - 1]
            assert f'"{r["entry_key"]}":' in line


def test_rs_fixture_list_matches_source_section():
    src = (gt.REPO_ROOT / gt.TABLES["robocasa"][0]).read_text().splitlines()
    start = next(i for i, l in enumerate(src) if "── Fixtures ──" in l)
    keys = {l.strip().split('"')[1] for l in src[start:] if l.startswith('    "') and l.rstrip().endswith("{")}
    assert keys == set(gt.RS_FIXTURES) | {"fixture_default"}


def test_non_literal_table_raises(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("OBJECT_PARAMS = {'a': {'mechanical': {'damage_threshold': float('1')}}}\n")
    with pytest.raises(ValueError, match="t.py"):
        gt.load_table("robocasa", path=p)


def test_float_roundtrip():
    for b in ("robocasa", "omnigibson"):
        for r in gt.export(b):
            for c in ("damage_threshold", "damage_scale", "impact_damage_sensitivity", "qs_damage_sensitivity"):
                if r[c] is not None:
                    assert float(gt.format_cell(r[c])) == r[c]


# ── Annotations: block groups, lockbox ──────────────────────────────────

MECH = ("damage_threshold", "damage_scale", "impact_damage_sensitivity", "qs_damage_sensitivity")


def test_every_target_annotated():
    for b in ("robocasa", "omnigibson"):
        assert all(r["block_group"] for r in gt.export(b) if r["is_target"])


def test_identical_tuples_share_block_group():
    for b in ("robocasa", "omnigibson"):
        seen = {}
        for r in (r for r in gt.export(b) if r["is_target"]):
            t = tuple(r[c] for c in MECH)
            assert seen.setdefault(t, r["block_group"]) == r["block_group"], (b, r["entry_key"])


def test_semantic_pairs_share_block_group():
    pairs = {
        "robocasa": [("cup", "mug"), ("wine", "liquor"), ("cabinet", "drawer")],
        "omnigibson": [("bottle_of_wine", "bottle_of_beer"), ("wineglass", "water_glass"),
                       ("top_cabinet", "bookcase"), ("bottom_cabinet", "bookcase"),
                       ("book", "comic_book"), ("floor_lamp", "table_lamp")],
    }
    for b, ps in pairs.items():
        g = {r["entry_key"]: r["block_group"] for r in gt.export(b)}
        assert all(g[x] == g[y] for x, y in ps), b


def test_lockbox():
    assert {r["entry_key"] for r in gt.export("robocasa") if r["lockbox"]} == {"banana", "jam", "milk"}
    assert {r["entry_key"] for r in gt.export("omnigibson") if r["lockbox"]} == {
        "box_of_crackers", "mirror", "openable_window"}


def test_lockbox_rows_are_singletons():
    for b in ("robocasa", "omnigibson"):
        rows = [r for r in gt.export(b) if r["is_target"]]
        for r in (r for r in rows if r["lockbox"]):
            assert sum(x["block_group"] == r["block_group"] for x in rows) == 1
            assert r["has_link_overrides"] is False


def test_annotation_unknown_key_raises():
    ann = {"material": None, "block_group": "x", "lockbox": False}
    with pytest.raises(ValueError, match="no_such_key"):
        gt.export("robocasa", annotations={("robocasa", "no_such_key"): ann})
    with pytest.raises(ValueError, match="Panda"):
        gt.export("robocasa", annotations={("robocasa", "Panda"): ann})


def test_bad_lockbox_value_raises(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("backend,entry_key,material,block_group,lockbox\nrobocasa,egg,,egg,yes\n")
    with pytest.raises(ValueError, match="lockbox"):
        gt.load_annotations(p)


# ── Kwarg contract: table keys are real evaluator __init__ parameters ───

def _init_params(rel_path, class_name):
    tree = ast.parse((gt.REPO_ROOT / rel_path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return {a.arg for a in init.args.args + init.args.kwonlyargs} - {"self", "entity"}


_CORE = {
    "mechanical": ("damagesim/core/evaluators/mechanical.py", "MechanicalDamageEvaluator"),
    "thermal": ("damagesim/core/evaluators/thermal.py", "ThermalDamageEvaluator"),
    "electrical": ("damagesim/core/evaluators/electrical.py", "ElectricalDamageEvaluator"),
}
_BACKEND = {
    ("robocasa", "mechanical"): ("damagesim/robosuite/evaluators/mechanical.py", "RSMechanicalDamageEvaluator"),
    ("omnigibson", "mechanical"): ("damagesim/omnigibson/evaluators/mechanical.py", "OGMechanicalDamageEvaluator"),
    ("omnigibson", "thermal"): ("damagesim/omnigibson/evaluators/thermal.py", "OGThermalDamageEvaluator"),
    ("omnigibson", "electrical"): ("damagesim/omnigibson/evaluators/electrical.py", "OGElectricalDamageEvaluator"),
}


def _allowed(backend, evaluator):
    return _init_params(*_CORE[evaluator]) | _init_params(*_BACKEND[(backend, evaluator)])


def test_table_kwargs_exist_in_evaluator_signatures():
    for b in ("robocasa", "omnigibson"):
        table, _ = gt.load_table(b)
        for key, entry in table.items():
            for ev in ("mechanical", "thermal", "electrical"):
                if ev in entry:
                    assert set(entry[ev]) <= _allowed(b, ev), (b, key, ev, set(entry[ev]) - _allowed(b, ev))
    assert not {"impact_sensitivity"} <= _allowed("robocasa", "mechanical")


# ── CSV writer, CLI, --check ────────────────────────────────────────────

def test_committed_csv_is_current():
    assert gt.main(["--check"]) == 0


def test_check_detects_stale_csv(tmp_path, capsys):
    p = tmp_path / "gt.csv"
    assert gt.main(["--out", str(p)]) == 0
    p.write_text(p.read_text().replace("15.0", "16.0", 1))
    assert gt.main(["--out", str(p), "--check"]) == 1
    assert "differs" in capsys.readouterr().err


def test_check_missing_file_fails(tmp_path):
    assert gt.main(["--out", str(tmp_path / "missing.csv"), "--check"]) == 1


def test_csv_uses_lf(tmp_path):
    p = tmp_path / "gt.csv"
    gt.main(["--out", str(p)])
    assert b"\r\n" not in p.read_bytes()


def test_csv_header_and_rows(tmp_path):
    p = tmp_path / "gt.csv"
    gt.main(["--out", str(p)])
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == gt.COLUMNS and len(rows) == 84


def test_cli_from_other_cwd(tmp_path):
    out = tmp_path / "gt.csv"
    r = subprocess.run([sys.executable, "-m", "damagesim.llm_params.ground_truth", "--out", str(out)],
                       cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(gt.REPO_ROOT)})
    assert r.returncode == 0 and out.exists()


# ── Independent cross-check (replaces Notion P0.1, hand transcription) ──

def test_csv_matches_imported_tables():
    # Real module import + this test's own column mapping; shares no code with the exporter.
    pytest.importorskip("numpy")
    from damagesim.robosuite.params.damage_params import OBJECT_PARAMS
    from damagesim.omnigibson.params.damage_params import PARAMS
    tables = {"robocasa": OBJECT_PARAMS, "omnigibson": PARAMS}
    mapping = {
        "damage_threshold": ("mechanical", "damage_threshold"),
        "damage_scale": ("mechanical", "damage_scale"),
        "impact_damage_sensitivity": ("mechanical", "impact_damage_sensitivity"),
        "qs_damage_sensitivity": ("mechanical", "qs_damage_sensitivity"),
        "heating_threshold": ("thermal", "heating_threshold"),
        "cooling_threshold": ("thermal", "cooling_threshold"),
        "thermal_scale": ("thermal", "scale"),
        "electrical_threshold": ("electrical", "damage_threshold"),
        "electrical_scale": ("electrical", "scale"),
    }
    with (gt.REPO_ROOT / "data/ground_truth_params.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {(r["backend"], r["entry_key"]) for r in rows} == {(b, k) for b, t in tables.items() for k in t}
    for r in rows:
        entry = tables[r["backend"]][r["entry_key"]]
        assert r["damage_evaluators"] == ";".join(entry["damage_evaluators"])
        for col, (block, key) in mapping.items():
            expected = entry.get(block, {}).get(key)
            if expected is None:
                assert r[col] == "", (r["backend"], r["entry_key"], col)
            else:
                assert float(r[col]) == expected, (r["backend"], r["entry_key"], col)
