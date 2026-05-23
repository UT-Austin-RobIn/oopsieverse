# OopsieBench

OopsieBench groups **damage-aware** task environments across simulators. Below, **OmniGibson / Behavior-1K** tasks correspond to the Python modules under `oopsiebench/envs/behavior1k/`. **RoboCasa (Robosuite)** tasks are registered in `oopsiebench/envs/registry.py`.

!!! tip "Videos and Git LFS"

    **RoboCasa:** each **Play** control opens an MP4 in a **GLightbox** overlay. Add clips under `docs/assets/videos/robocasa/<task_id>.mp4` (tracked with **Git LFS**; see repo `.gitattributes` and `docs/assets/videos/robocasa/VIDEOS.txt`). Until a file exists, the button still appears; fix broken links by adding the matching MP4.

    **Behavior-1K:** placeholders for now; you can mirror the same pattern under `docs/assets/videos/behavior1k/` later (add a matching LFS line in `.gitattributes` if you do).

## OmniGibson — Behavior-1K tasks

| Task ID | Video | Notes |
|---------|-------|-------|
| `add_firewood` | — | |
| `default` | — | Default B1K teleop scene (**Rs_int**) with broad damage tracking |
| `food_in_microwave` | — | |
| `open_drawer` | — | |
| `open_single_door` | — | |
| `pour_water` | — | |
| `shelve_item` | — | |
| `turn_on_stove` | — | |

## RoboCasa (Robosuite) — registered tasks

| Task ID | Video | Notes |
|---------|-------|-------|
| `pick_egg` | <a href="../assets/videos/robocasa/pick_egg.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `serve_pastry` | <a href="../assets/videos/robocasa/serve_pastry.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `open_single_door` | <a href="../assets/videos/robocasa/open_single_door.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `turn_on_faucet` | <a href="../assets/videos/robocasa/turn_on_faucet.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `turn_on_microwave` | <a href="../assets/videos/robocasa/turn_on_microwave.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `turn_on_stove` | <a href="../assets/videos/robocasa/turn_on_stove.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `open_drawer` | <a href="../assets/videos/robocasa/open_drawer.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `close_drawer` | <a href="../assets/videos/robocasa/close_drawer.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `place_plate` | <a href="../assets/videos/robocasa/place_plate.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `counter_to_microwave` | <a href="../assets/videos/robocasa/counter_to_microwave.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `prepare_coffee` | <a href="../assets/videos/robocasa/prepare_coffee.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `shelve_item` | <a href="../assets/videos/robocasa/shelve_item.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `prepare_breakfast` | <a href="../assets/videos/robocasa/prepare_breakfast.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `dishes_to_sink` | <a href="../assets/videos/robocasa/dishes_to_sink.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `nav_lift_bowl` | <a href="../assets/videos/robocasa/nav_lift_bowl.mp4" class="glightbox md-button" data-type="video">Play</a> | |
| `wipe_counter` | <a href="../assets/videos/robocasa/wipe_counter.mp4" class="glightbox md-button" data-type="video">Play</a> | |

Source for RoboCasa IDs: [`oopsiebench/envs/registry.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/oopsiebench/envs/registry.py).
