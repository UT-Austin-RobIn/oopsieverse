#!/usr/bin/env python3
"""Minimal SpaceMouse read test using pyspacemouse."""

import argparse
import sys
import time

import pyspacemouse


def format_state(state) -> str:
    return (
        f"x={state.x:+.3f} y={state.y:+.3f} z={state.z:+.3f} "
        f"roll={state.roll:+.3f} pitch={state.pitch:+.3f} yaw={state.yaw:+.3f} "
        f"buttons={list(state.buttons)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and print SpaceMouse values.")
    parser.add_argument(
        "--hz",
        type=float,
        default=20.0,
        help="Polling frequency in Hz (default: 20).",
    )
    args = parser.parse_args()

    dt = 1.0 / max(args.hz, 1e-3)

    try:
        device = pyspacemouse.open()
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Failed to open SpaceMouse: {exc}")
        return 1

    if not device:
        print("[ERROR] pyspacemouse.open() returned no device")
        return 1

    print("SpaceMouse opened. Move cap / press buttons. Ctrl+C to stop.")
    try:
        while True:
            state = device.read()
            if state is not None:
                print(format_state(state))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            device.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
