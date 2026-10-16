"""
Export the hand-tuned DamageSim parameter tables to one CSV (Notion task D0.3).

Reads ``OBJECT_PARAMS`` (RoboCasa) and ``PARAMS`` (OmniGibson) with ``ast``
instead of importing them, so this runs with the standard library alone, on
any Python >= 3.9, without numpy or a simulator. The tables must stay pure
literals; a computed value raises ``ValueError`` instead of exporting
something wrong.

    python -m damagesim.llm_params.ground_truth [--out PATH] [--check]
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TABLES = {
    "robocasa": ("damagesim/robosuite/params/damage_params.py", "OBJECT_PARAMS"),
    "omnigibson": ("damagesim/omnigibson/params/damage_params.py", "PARAMS"),
}

RS_FIXTURES = frozenset({
    "microwave", "cabinet", "drawer", "coffee_machine", "counter",
    "fridge", "stove", "sink", "dishwasher", "stool",
})

COLUMNS = [
    "backend", "entry_key", "kind", "is_target", "display_name", "damage_evaluators",
    "damage_threshold", "damage_scale", "impact_damage_sensitivity", "qs_damage_sensitivity",
    "has_link_overrides", "has_damageable_links",
    "heating_threshold", "cooling_threshold", "thermal_scale",
    "electrical_threshold", "electrical_scale", "electrical_water_system", "electrical_link_thresholds",
    "material", "block_group", "lockbox",
    "source_file", "source_line", "notes",
]

ANNOTATIONS_PATH = REPO_ROOT / "data/annotations.csv"
DEFAULT_OUT = REPO_ROOT / "data/ground_truth_params.csv"

_OVERRIDE_KEYS = ("link_config_overrides", "part_config_overrides")
_TARGET_KINDS = {"object", "fixture", "category"}


def _parse(backend: str, path: Path | None) -> tuple[Path, ast.Dict]:
    rel, name = TABLES[backend]
    path = Path(path) if path is not None else REPO_ROOT / rel
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if not isinstance(node.value, ast.Dict):
                raise ValueError(f"{path}: {name} is not a dict literal")
            return path, node.value
    raise ValueError(f"{path}: no top-level assignment to {name}")


def load_table(backend: str, path: Path | None = None) -> tuple[dict, dict[str, int]]:
    """Return the table as a dict and a map entry key -> source line of that key."""
    path, node = _parse(backend, path)
    try:
        table = ast.literal_eval(node)
    except ValueError as exc:
        raise ValueError(f"{path}: {TABLES[backend][1]} is not a pure literal: {exc}") from None
    lines = {k.value: k.lineno for k in node.keys}
    return table, lines


def _kind(backend: str, key: str) -> str:
    if backend == "robocasa":
        if key in ("default", "fixture_default"):
            return "default"
        if key in ("Panda", "PandaOmron"):
            return "robot"
        return "fixture" if key in RS_FIXTURES else "object"
    if key == "default":
        return "default"
    return "robot" if key == "agent" else "category"


def _misnested_override_line(entry_node: ast.Dict) -> int | None:
    for k in entry_node.keys:
        if isinstance(k, ast.Constant) and k.value in _OVERRIDE_KEYS:
            return k.lineno
    return None


def load_annotations(path: Path = ANNOTATIONS_PATH) -> dict[tuple[str, str], dict]:
    """Read the hand-edited annotations: (backend, entry_key) -> material, block_group, lockbox."""
    out = {}
    with open(path, newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=2):
            lockbox = rec["lockbox"].strip().lower()
            if lockbox not in ("true", "false"):
                raise ValueError(f"{path}:{i}: lockbox must be true or false, got {rec['lockbox']!r}")
            out[(rec["backend"], rec["entry_key"])] = {
                "material": rec["material"] or None,
                "block_group": rec["block_group"] or None,
                "lockbox": lockbox == "true",
            }
    return out


def export(backend: str, annotations: dict | None = None) -> list[dict]:
    """One row per table key, in source order, with typed values (None = missing).

    ``annotations`` defaults to ``data/annotations.csv``; every annotated key must
    be a target row of this backend's table.
    """
    if annotations is None:
        annotations = load_annotations()
    table, lines = load_table(backend)
    _, node = _parse(backend, None)
    entry_nodes = {k.value: v for k, v in zip(node.keys, node.values)}
    source_file = TABLES[backend][0]

    targets = {k for k in table if _kind(backend, k) in _TARGET_KINDS}
    for b, key in annotations:
        if b == backend and key not in targets:
            raise ValueError(f"annotations.csv: {backend}/{key} is not a target row in {source_file}")

    rows = []
    for key, entry in table.items():
        kind = _kind(backend, key)
        mech = entry.get("mechanical", {})
        thermal = entry.get("thermal")
        elec = entry.get("electrical")
        notes = []

        misnested = _misnested_override_line(entry_nodes[key])
        if misnested is not None:
            notes.append(f"part_config_overrides misnested (ignored at runtime, {source_file}:{misnested})")
        if "robot_overrides" in entry:
            notes.append("robot_overrides for: " + ", ".join(sorted(entry["robot_overrides"])) + " (not exported)")

        link_thr = (elec or {}).get("link_thresholds")
        ann = annotations.get((backend, key), {})
        rows.append({
            "backend": backend,
            "entry_key": key,
            "kind": kind,
            "is_target": kind in _TARGET_KINDS,
            "display_name": key.replace("_", " "),
            "damage_evaluators": ";".join(entry.get("damage_evaluators", [])),
            "damage_threshold": mech.get("damage_threshold"),
            "damage_scale": mech.get("damage_scale"),
            "impact_damage_sensitivity": mech.get("impact_damage_sensitivity"),
            "qs_damage_sensitivity": mech.get("qs_damage_sensitivity"),
            "has_link_overrides": any(k in mech for k in _OVERRIDE_KEYS),
            "has_damageable_links": "damageable_links" in entry,
            "heating_threshold": thermal.get("heating_threshold") if thermal else None,
            "cooling_threshold": thermal.get("cooling_threshold") if thermal else None,
            "thermal_scale": thermal.get("scale") if thermal else None,
            "electrical_threshold": elec.get("damage_threshold") if elec else None,
            "electrical_scale": elec.get("scale") if elec else None,
            "electrical_water_system": elec.get("water_system_name") if elec else None,
            "electrical_link_thresholds": (
                json.dumps(link_thr, sort_keys=True, separators=(",", ":")) if link_thr else None
            ),
            "material": ann.get("material"),
            "block_group": ann.get("block_group"),
            "lockbox": ann.get("lockbox", False),
            "source_file": source_file,
            "source_line": lines[key],
            "notes": "; ".join(notes) or None,
        })
    return rows


def format_cell(value) -> str:
    """CSV text for one value: '' for None, true/false for bools, repr() for floats (exact round-trip)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def to_csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: format_cell(row[c]) for c in COLUMNS})
    return buf.getvalue()


def write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(to_csv_text(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV path (default: data/ground_truth_params.csv)")
    parser.add_argument("--check", action="store_true", help="fail if --out differs from the tables instead of writing it")
    args = parser.parse_args(argv)

    rows = export("robocasa") + export("omnigibson")
    text = to_csv_text(rows)
    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist; run without --check to generate it", file=sys.stderr)
            return 1
        with open(args.out, newline="") as f:
            current = f.read()
        if current != text:
            old, new = current.splitlines(), text.splitlines()
            line = next((i for i, (a, b) in enumerate(zip(old, new), start=1) if a != b), min(len(old), len(new)) + 1)
            print(f"{args.out} differs from the tables; first difference at line {line}. "
                  "Regenerate with: python -m damagesim.llm_params.ground_truth", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date ({len(rows)} rows)")
        return 0
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
