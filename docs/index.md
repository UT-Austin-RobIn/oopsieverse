<p align="center">
  <img
    src="assets/images/oopsieverse_logo.png"
    alt="OopsieVerse logo"
    loading="lazy"
    class="docs-brand-logo"
  />
</p>

OopsieVerse is a **damage-aware**, **simulator-agnostic** framework and benchmark for evaluating and learning safer policies. OopsieVerse models damage as a unified signal called **health**.

## Documentation

- **[Install](install.md)** — Clone the repo, set up conda environments, and install BEHAVIOR-1K (OmniGibson) and/or RoboCasa (MuJoCo).
- **[Quickstart](quickstart.md)** — Download pre-collected safe/unsafe demos and run teleop or playback with health overlays.
- **[DamageSim](damagesim.md)** — Core damage API: health evaluators, the `DamageableMixin` contract, and simulator integration.
- **[OopsieBench](oopsiebench.md)** — Benchmark task catalog with safe vs. unsafe demo videos for both simulators.
- **[Troubleshooting](troubleshooting.md)** — Common setup fixes (CUDA/PyTorch, SpaceMouse udev, and related issues).

## Key Features

- **Unified damage signal** — a single continuous *health* value per object/link, aggregating mechanical, thermal, and electrical damage.
- **Simulator-agnostic** — one consistent damage API across BEHAVIOR-1K (OmniGibson) and RoboCasa (MuJoCo).
- **Real-time safety feedback** — live health bars and damage-based object coloring during teleoperation, so collectors can gather safer demonstrations.
- **Benchmark tasks** — a growing set of contact-, heat-, and fluid-rich manipulation tasks in realistic kitchens and homes.
- **End-to-end pipeline** — teleoperate → playback → render health-overlay videos and compute per-object safety metrics.

