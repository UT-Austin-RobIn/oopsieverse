#!/usr/bin/env python3
"""
Utilities for editing OmniGibson / oopsieverse trajectory HDF5 files.

Demos live under ``data/demo_0``, ``data/demo_1``, … with group-level datasets
and attrs (see ``scripts/inspect_hdf5.py``).

Usage:
    # Delete demos 1 and 3 (renumber remaining demos to demo_0, demo_1, …)
    python scripts/modify_hdf5.py delete path/to/file.hdf5 --demos 1 3

    # Delete in place (writes via a temp file, then replaces the original)
    python scripts/modify_hdf5.py delete path/to/file.hdf5 --demos 2 --in-place

    # Merge two files into one (demos from A, then demos from B, renumbered)
    python scripts/modify_hdf5.py combine path/a.hdf5 path/b.hdf5 -o path/merged.hdf5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from typing import Iterable, List, Sequence, Union

import h5py

DemoId = Union[int, str]

_ATTRS_RECOMPUTED = frozenset({"n_episodes", "n_steps"})


def _normalize_demo_key(demo_id: DemoId) -> str:
    if isinstance(demo_id, str):
        if demo_id.startswith("demo_"):
            return demo_id
        if demo_id.isdigit():
            return f"demo_{demo_id}"
        raise ValueError(f"Invalid demo id string: {demo_id!r}")
    if isinstance(demo_id, int):
        if demo_id < 0:
            raise ValueError(f"Demo index must be non-negative, got {demo_id}")
        return f"demo_{demo_id}"
    raise TypeError(f"demo_id must be int or str, got {type(demo_id)}")


def _list_demo_keys(data_grp: h5py.Group) -> List[str]:
    keys = [k for k in data_grp.keys() if k.startswith("demo_") and isinstance(data_grp[k], h5py.Group)]
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def _count_demo_steps(demo_grp: h5py.Group) -> int:
    if "num_samples" in demo_grp.attrs:
        return int(demo_grp.attrs["num_samples"])
    if "state" in demo_grp and hasattr(demo_grp["state"], "shape"):
        return int(demo_grp["state"].shape[0])
    for key in demo_grp.keys():
        obj = demo_grp[key]
        if isinstance(obj, h5py.Dataset) and len(obj.shape) > 0:
            return int(obj.shape[0])
    return 0


def _copy_group_attrs(src: h5py.Group, dst: h5py.Group, *, skip: frozenset[str] = _ATTRS_RECOMPUTED) -> None:
    for name, value in src.attrs.items():
        if name in skip:
            continue
        dst.attrs[name] = value


def _update_data_metadata(data_grp: h5py.Group) -> None:
    demo_keys = _list_demo_keys(data_grp)
    data_grp.attrs["n_episodes"] = len(demo_keys)
    data_grp.attrs["n_steps"] = sum(_count_demo_steps(data_grp[k]) for k in demo_keys)


def _write_demos_to_file(
    *,
    source_files: Sequence[tuple[h5py.File, Sequence[str]]],
    output_path: str,
    metadata_from: h5py.Group,
) -> int:
    """Copy selected demos from one or more open HDF5 files into ``output_path``."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as out_f:
        out_data = out_f.create_group("data")
        _copy_group_attrs(metadata_from, out_data)

        out_idx = 0
        for src_f, demo_keys in source_files:
            src_data = src_f["data"]
            for demo_key in demo_keys:
                if demo_key not in src_data:
                    raise KeyError(
                        f"Demo {demo_key!r} not found in {src_f.filename}. "
                        f"Available: {_list_demo_keys(src_data)}"
                    )
                src_data.copy(demo_key, out_data, f"demo_{out_idx}")
                out_idx += 1

        _update_data_metadata(out_data)

    return out_idx


def delete_demos(
    hdf5_path: str,
    demo_ids: Sequence[DemoId],
    output_path: str | None = None,
) -> str:
    """
    Remove specific demos from an HDF5 file.

    Remaining demos are renumbered to ``demo_0`` … ``demo_{N-1}`` so playback
    scripts that iterate ``range(n_episodes)`` keep working.

    Args:
        hdf5_path: Input HDF5 path.
        demo_ids: Demo indices (``0``, ``1``, …) or keys (``"demo_0"``).
        output_path: Output path. If ``None``, overwrites ``hdf5_path`` via a temp file.

    Returns:
        Path to the written HDF5 file.
    """
    if not demo_ids:
        raise ValueError("demo_ids must be non-empty")

    to_remove = {_normalize_demo_key(d) for d in demo_ids}
    out_path = output_path or hdf5_path
    in_place = output_path is None or os.path.abspath(output_path) == os.path.abspath(hdf5_path)

    with h5py.File(hdf5_path, "r") as src_f:
        src_data = src_f["data"]
        if "data" not in src_f:
            raise KeyError(f"No 'data' group in {hdf5_path}")

        all_keys = _list_demo_keys(src_data)
        missing = to_remove - set(all_keys)
        if missing:
            raise KeyError(f"Demo(s) not in file: {sorted(missing)}. Available: {all_keys}")

        keep_keys = [k for k in all_keys if k not in to_remove]
        if not keep_keys:
            raise ValueError("Cannot delete all demos; at least one demo must remain.")

        if in_place:
            fd, tmp_path = tempfile.mkstemp(suffix=".hdf5", dir=os.path.dirname(os.path.abspath(hdf5_path)) or ".")
            os.close(fd)
        else:
            tmp_path = out_path

        n_written = _write_demos_to_file(
            source_files=[(src_f, keep_keys)],
            output_path=tmp_path,
            metadata_from=src_data,
        )

    if in_place:
        shutil.move(tmp_path, hdf5_path)
        out_path = hdf5_path

    print(
        f"[modify_hdf5] Deleted {len(to_remove)} demo(s) from {hdf5_path} "
        f"→ {n_written} demo(s) in {out_path}"
    )
    return out_path


def combine_hdf5_files(
    hdf5_path_a: str,
    hdf5_path_b: str,
    output_path: str,
    *,
    require_matching_config: bool = False,
) -> str:
    """
    Combine demos from two HDF5 files into a single file.

    All demos from ``hdf5_path_a`` are copied first, then all demos from
    ``hdf5_path_b``, renumbered to ``demo_0`` … ``demo_{N-1}``.

    Top-level ``data`` attrs (``config``, ``scene_file``, …) are taken from file A.
    ``n_episodes`` and ``n_steps`` are recomputed.

    Args:
        hdf5_path_a: First input HDF5 (demos copied first).
        hdf5_path_b: Second input HDF5 (demos appended after A).
        output_path: Output HDF5 path.
        require_matching_config: If True, raise when ``config`` attrs differ.

    Returns:
        Path to the written HDF5 file.
    """
    with h5py.File(hdf5_path_a, "r") as fa, h5py.File(hdf5_path_b, "r") as fb:
        if "data" not in fa or "data" not in fb:
            raise KeyError("Both HDF5 files must contain a top-level 'data' group.")

        data_a, data_b = fa["data"], fb["data"]
        keys_a = _list_demo_keys(data_a)
        keys_b = _list_demo_keys(data_b)
        if not keys_a:
            raise ValueError(f"No demos found in {hdf5_path_a}")
        if not keys_b:
            raise ValueError(f"No demos found in {hdf5_path_b}")

        if require_matching_config and "config" in data_a.attrs and "config" in data_b.attrs:
            if data_a.attrs["config"] != data_b.attrs["config"]:
                raise ValueError(
                    "config attrs differ between input files. "
                    "Use require_matching_config=False to merge anyway."
                )
        elif "config" in data_a.attrs and "config" in data_b.attrs:
            if data_a.attrs["config"] != data_b.attrs["config"]:
                print(
                    "[modify_hdf5] Warning: config attrs differ between inputs; "
                    "using config from the first file."
                )

        n_written = _write_demos_to_file(
            source_files=[(fa, keys_a), (fb, keys_b)],
            output_path=output_path,
            metadata_from=data_a,
        )

    print(
        f"[modify_hdf5] Combined {len(keys_a)} + {len(keys_b)} demo(s) "
        f"→ {n_written} demo(s) in {output_path}"
    )
    return output_path


def _parse_demo_ids(values: Iterable[str]) -> List[DemoId]:
    demo_ids: List[DemoId] = []
    for v in values:
        demo_ids.append(int(v) if v.isdigit() else v)
    return demo_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Modify oopsieverse / OmniGibson trajectory HDF5 files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_delete = sub.add_parser("delete", help="Delete specific demos from an HDF5 file.")
    p_delete.add_argument("hdf5_path", help="Input HDF5 path.")
    p_delete.add_argument(
        "--demos",
        nargs="+",
        required=True,
        metavar="ID",
        help="Demo indices (e.g. 0 2) or keys (e.g. demo_1).",
    )
    p_delete.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: overwrite input with --in-place, else required).",
    )
    p_delete.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file (default when -o is omitted).",
    )

    p_combine = sub.add_parser("combine", help="Combine demos from two HDF5 files.")
    p_combine.add_argument("hdf5_path_a", help="First HDF5 (demos copied first).")
    p_combine.add_argument("hdf5_path_b", help="Second HDF5 (demos appended).")
    p_combine.add_argument("-o", "--output", required=True, help="Output HDF5 path.")
    p_combine.add_argument(
        "--require-matching-config",
        action="store_true",
        help="Fail if config attrs differ between inputs.",
    )

    args = parser.parse_args(argv)

    if args.command == "delete":
        if args.output is None and not args.in_place:
            # Default to in-place when no output given (matches docstring).
            args.in_place = True
        if args.output is not None and args.in_place:
            parser.error("Use either --output or --in-place, not both.")
        delete_demos(
            args.hdf5_path,
            _parse_demo_ids(args.demos),
            output_path=args.output,
        )
    elif args.command == "combine":
        combine_hdf5_files(
            args.hdf5_path_a,
            args.hdf5_path_b,
            args.output,
            require_matching_config=args.require_matching_config,
        )
    else:
        parser.error(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
