# Learning

Train a conditional flow-matching (CFM) policy on OopsieVerse paper-experiment tasks.

## Paper tasks

| Task | Simulator | Description | Teleop demos |
| :--- | :-------- | :---------- | :----------- |
| `shelve_item` | BEHAVIOR-1K | Shelve an item among fragile objects (mechanical) | 45 without + 45 with live feedback (90 total) |
| `add_firewood` | BEHAVIOR-1K | Place firewood in the fireplace (mechanical + thermal) | 30 without + 30 with live feedback (60 total) |
| `pick_egg` | RoboCasa | Pick up an egg without crushing it (mechanical) | 30 without + 30 with live feedback (60 total) |
| `wipe_counter` | RoboCasa | Wipe dirt on the counter with a sponge (mechanical) | 30 without + 30 with live feedback (60 total) |

Download the teleop splits with:

```bash
python scripts/download_demos.py --paper-demos
```

They land in `oopsiebench/demos/paper_demos/teleop_data/<task>/` (`without_live_feedback.hdf5`, `with_live_feedback.hdf5`, `all_data.hdf5`).

## Playback before training

CFM training expects **playback** HDF5s (observations, health, cameras), not raw teleop state dumps. After downloading paper demos, run playback for the split you want to train on, then point the config `data_path` at that playback file (see [Quickstart](quickstart.md) for playback commands).

## Train

```bash
python -m learning.train_eval.cfm_trainer --config learning/configs/add_firewood.yaml
python -m learning.train_eval.cfm_trainer --config learning/configs/shelve_item.yaml
python -m learning.train_eval.cfm_trainer --config learning/configs/pick_egg.yaml
python -m learning.train_eval.cfm_trainer --config learning/configs/wipe_counter.yaml
```

Configs live under `learning/configs/`. Override fields on the CLI if needed (for example `--device cuda` or `--data-path path/to/playback.hdf5`).
