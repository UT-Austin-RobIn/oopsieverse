"""
Visualization helpers for DamageSim.

Contains:
* **Live health bars** — real-time matplotlib display during teleoperation.
* **Video saving** — ``save_rgb_camera_video`` and
  ``save_rgb_health_video_with_overlay`` used by the unified playback script.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.animation as animation

try:
    import matplotlib
    matplotlib.use("TkAgg")  # Non-blocking backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════
# Live health-bar helpers (matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def setup_live_health_bars(
    object_names: List[str],
    obj_display_names: Optional[Dict[str, str]] = None,
) -> Tuple["Figure", "Axes", Dict]:
    """
    Create a draggable HUD-style matplotlib window with health bars.

    The window has a dark background, a "Health Monitor" title, and one
    colour-coded bar per object.  It is borderless and can be dragged
    with the mouse.

    Returns:
        ``(fig, ax, bars_dict)`` where *bars_dict* maps each object name
        to a dict of patch / text handles used by :func:`update_live_health_bars`.
    """
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for health visualization")

    from matplotlib.patches import Rectangle

    plt.ion()

    display_map = dict(OBJ_NAME_DISPLAY_NAME_MAPPING)
    if obj_display_names:
        display_map.update(obj_display_names)

    n_objects = len(object_names)
    bar_height = 30.0
    bar_spacing = 10.0
    header_height = 60.0
    padding = 15.0
    window_width = 600.0
    window_height = header_height + n_objects * (bar_height + bar_spacing) + padding * 2

    fig, ax = plt.subplots(figsize=(window_width / 100, window_height / 100))

    # Make the window borderless and draggable (TkAgg specific)
    try:
        window = fig.canvas.manager.window
        window.overrideredirect(True)

        def _start_move(event):
            window._drag_start_x = event.x
            window._drag_start_y = event.y

        def _do_move(event):
            x = window.winfo_pointerx() - window._drag_start_x
            y = window.winfo_pointery() - window._drag_start_y
            window.geometry(f"+{x}+{y}")

        window.bind("<Button-1>", _start_move)
        window.bind("<B1-Motion>", _do_move)
        window.bind("<Escape>", lambda e: window.destroy())
    except Exception:
        pass

    fig.patch.set_facecolor("#1A1A1A")
    ax.set_facecolor("#1A1A1A")
    ax.axis("off")
    ax.set_xlim(0, window_width)
    ax.set_ylim(0, window_height)

    # Title
    ax.text(
        window_width / 2, window_height - 35,
        "Health Monitor",
        fontsize=18, color="#FFFFFF", weight="bold",
        verticalalignment="center", horizontalalignment="center",
        family="sans-serif",
    )
    ax.add_patch(Rectangle(
        (window_width / 2 - 80, window_height - 50), 160, 2,
        facecolor="#4CAF50", edgecolor="none", linewidth=0,
        zorder=1, alpha=0.6,
    ))

    label_width = 160.0
    bar_width = 280.0
    gap_after_label = 20.0
    gap_after_bar = 20.0
    bar_x_start = label_width + gap_after_label
    label_x = padding
    value_x = bar_x_start + bar_width + gap_after_bar

    bars_dict: Dict[str, Dict] = {}

    for idx, obj_name in enumerate(object_names):
        y_pos = window_height - header_height - (idx + 1) * (bar_height + bar_spacing) - padding / 2

        bg_bar = Rectangle(
            (bar_x_start, y_pos), bar_width, bar_height,
            facecolor="#0D0D0D", edgecolor="#2A2A2A", linewidth=2.5, zorder=1,
        )
        ax.add_patch(bg_bar)

        glow_bar = Rectangle(
            (bar_x_start, y_pos + bar_height - 3), bar_width, 3,
            facecolor="#333333", edgecolor="none", linewidth=0, alpha=0.4, zorder=2,
        )
        ax.add_patch(glow_bar)

        shadow_bar = Rectangle(
            (bar_x_start + 2, y_pos + 2), bar_width - 4, bar_height - 4,
            facecolor="none", edgecolor="#000000", linewidth=1, alpha=0.5, zorder=2,
        )
        ax.add_patch(shadow_bar)

        fg_bar = Rectangle(
            (bar_x_start + 3, y_pos + 3), bar_width - 6, bar_height - 6,
            facecolor="#4CAF50", edgecolor="none", linewidth=0, zorder=3, alpha=0.95,
        )
        ax.add_patch(fg_bar)

        display_name = display_map.get(obj_name, obj_name)
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."

        label_text = ax.text(
            label_x, y_pos + bar_height / 2,
            display_name,
            fontsize=12, color="#E8E8E8", weight="normal",
            verticalalignment="center", horizontalalignment="left",
            family="monospace",
        )

        value_text = ax.text(
            value_x, y_pos + bar_height / 2,
            "100.0",
            fontsize=12, color="#FFFFFF", weight="bold",
            verticalalignment="center", horizontalalignment="left",
            family="sans-serif",
        )

        bars_dict[obj_name] = {
            "foreground_bar": fg_bar,
            "label_text": label_text,
            "value_text": value_text,
            "bar_width": bar_width - 6,
        }

    plt.tight_layout()
    plt.show(block=False)

    return fig, ax, bars_dict


def update_live_health_bars(
    fig: "Figure",
    ax: "Axes",
    bars_dict: Dict[str, Dict],
    current_health: Dict[str, float],
    object_names: List[str],
) -> bool:
    """
    Update the live HUD health bars with new values.

    Returns:
        True if the figure is still open, False if it was closed.
    """
    if not HAS_MPL:
        return False

    if not plt.fignum_exists(fig.number):
        return False

    for obj_name in object_names:
        if obj_name not in bars_dict:
            continue
        health = max(0.0, min(100.0, current_health.get(obj_name, 100.0)))
        bar_info = bars_dict[obj_name]
        fg_bar = bar_info["foreground_bar"]
        value_text = bar_info["value_text"]
        label_text = bar_info["label_text"]
        full_width = bar_info["bar_width"]
        health_width = (health / 100.0) * full_width

        if health == 0:
            fg_bar.set_facecolor("none")
            fg_bar.set_width(0)
            value_color = "#666666"
        elif health >= 80:
            fg_bar.set_facecolor("#4CAF50")
            fg_bar.set_width(health_width)
            value_color = "#E8F5E9"
        elif health >= 60:
            fg_bar.set_facecolor("#FFC107")
            fg_bar.set_width(health_width)
            value_color = "#FFF9C4"
        elif health >= 40:
            fg_bar.set_facecolor("#FF9800")
            fg_bar.set_width(health_width)
            value_color = "#FFE0B2"
        elif health >= 20:
            fg_bar.set_facecolor("#F44336")
            fg_bar.set_width(health_width)
            value_color = "#FFCDD2"
        else:
            fg_bar.set_facecolor("#D32F2F")
            fg_bar.set_width(health_width)
            value_color = "#EF9A9A"

        value_text.set_text(f"{health:.1f}")
        value_text.set_color(value_color)
        label_text.set_color("#888888" if health == 0 else "#E8E8E8")

    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════
# Video saving utilities
# ═══════════════════════════════════════════════════════════════════════

# Display-name mapping shared across tasks
OBJ_NAME_DISPLAY_NAME_MAPPING: Dict[str, str] = {
    "box_of_crackers": "Crackers Box",
    "book": "Paper Bag",
    "bottle_of_wine": "Wine Bottle",
    "wineglass": "Wine Glass",
    "bottle_of_beer": "Beer Bottle",
    "franka0": "Robot",
    "laptop": "Laptop",
}


def save_rgb_camera_video(
    output_video_path: str,
    imgs: np.ndarray,
    fps: int = 30,
) -> None:
    """
    Save an array of RGB images as an MP4 video (via AVI intermediate + ffmpeg).

    Args:
        output_video_path: Destination path **without** extension (or with
            ``.mp4`` — the function appends ``.avi`` / ``.mp4`` as needed).
        imgs: ``(T, H, W, 3)`` uint8 array in **RGB** order.
        fps: Frames per second.
    """
    if len(imgs) == 0:
        return

    # Normalise path: strip .mp4/.avi if caller already added one
    base, ext = os.path.splitext(output_video_path)
    if ext.lower() in (".mp4", ".avi"):
        base = base  # already stripped
    else:
        base = output_video_path

    avi_video = base + ".avi"
    mp4_video = base + ".mp4"

    he, we = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(avi_video, fourcc, fps, (we, he))
    for img in imgs:
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        vw.write(np.ascontiguousarray(bgr, dtype=np.uint8))
    vw.release()

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", avi_video,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
            mp4_video,
        ],
        check=True,
    )
    if os.path.exists(avi_video):
        os.remove(avi_video)


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Colour palette used by the health-bar overlay
_COLORS = {
    "bg": _hex_to_bgr("#1A1A1A"),
    "bg_bar": _hex_to_bgr("#0D0D0D"),
    "border": _hex_to_bgr("#2A2A2A"),
    "glow": _hex_to_bgr("#333333"),
    "green": _hex_to_bgr("#4CAF50"),
    "amber": _hex_to_bgr("#FFC107"),
    "orange": _hex_to_bgr("#FF9800"),
    "red": _hex_to_bgr("#F44336"),
    "dark_red": _hex_to_bgr("#D32F2F"),
    "text_white": _hex_to_bgr("#FFFFFF"),
    "text_light": _hex_to_bgr("#E8E8E8"),
    "text_gray": _hex_to_bgr("#888888"),
    "text_dark_gray": _hex_to_bgr("#666666"),
}


def save_rgb_health_video_with_overlay(
    output_video_path: str,
    imgs: np.ndarray,
    target_objects: List[str],
    health: Dict[str, Optional[np.ndarray]],
    position: str = "bottom_left",
    n_columns: int = 1,
    fps: int = 30,
    obj_display_names: Optional[Dict[str, str]] = None,
    layout: str = "column",
) -> None:
    """
    Save video with health bars overlaid directly on the RGB frames.

    Args:
        output_video_path: Destination path (appended with ``.mp4``).
        imgs: ``(T, H, W, 3)`` uint8 array in **RGB** order.
        target_objects: Object names whose health to display.
        health: Mapping ``obj_name → 1-D array of health values (0–100)``.
        position: Where to place the panel. Horizontal: ``"*_left"``, ``"*_right"``,
            ``"*_center"`` or ``"center"``. Vertical: ``"bottom_*"``, ``"top_*"``, or
            ``"center"`` (vertical+horizontal center). Examples: ``"bottom_left"``,
            ``"bottom_right"``, ``"bottom_center"``, ``"top_left"``, ``"top_right"``,
            ``"top_center"``, ``"center"``.
        n_columns: Number of columns for laying out health bars (1–3); used only when layout="column".
        fps: Frames per second.
        obj_display_names: Optional override for display names.
        layout: ``"column"`` = bars stacked in columns (default). ``"row"`` = all bars in one
            horizontal row, centered on the frame.
    """
    # Pick the first non-None health array to get the number of frames
    T = 0
    for obj in target_objects:
        h = health.get(obj)
        if h is not None:
            T = len(h)
            break
    if T == 0 or len(imgs) == 0:
        return

    display_map = dict(OBJ_NAME_DISPLAY_NAME_MAPPING)
    if obj_display_names:
        display_map.update(obj_display_names)

    n_columns = max(1, min(3, int(n_columns)))
    n_objects = len(target_objects)

    img_height, img_width = imgs[0].shape[:2]

    # ── Bar dimensions (scale with image size) ────────────────────────
    bar_height = max(25, int(img_height * 0.05))
    bar_spacing = max(8, int(img_height * 0.01))
    padding = max(15, int(img_width * 0.02))
    column_spacing = max(15, int(img_width * 0.02))
    row_spacing = max(15, int(img_width * 0.02))

    label_width = max(80, int(img_width * 0.13))
    bar_width = max(100, int(img_width * 0.1))
    gap_after_label = max(4, int(img_width * 0.012))
    gap_after_bar = max(8, int(img_width * 0.012))
    value_width = max(50, int(img_width * 0.06))
    font_size = max(0.4, min(1.5, img_width / 1600.0))

    # Parse position: vertical_top | vertical_bottom | vertical_center, horizontal_left | right | center
    position_lower = (position or "bottom_left").lower()
    if position_lower == "center":
        vertical, horizontal = "center", "center"
    else:
        parts = position_lower.split("_")
        if len(parts) >= 2:
            vertical = parts[0]   # top, bottom, center
            horizontal = parts[1] if len(parts) > 1 else "left"
        else:
            vertical, horizontal = "bottom", "left"

    use_row_layout = layout == "row"
    if use_row_layout:
        item_width = (
            label_width
            + gap_after_label
            + bar_width
            + gap_after_bar
            + value_width
        )
        panel_width = (
            n_objects * item_width
            + (n_objects - 1) * row_spacing
            + padding * 2
        )
        panel_height = bar_height + padding * 2
        panel_x = (img_width - panel_width) // 2
    else:
        objects_per_column = int(np.ceil(n_objects / n_columns))
        column_width = (
            label_width
            + gap_after_label
            + bar_width
            + gap_after_bar
            + value_width
        )
        panel_width = n_columns * column_width + (n_columns - 1) * column_spacing + 30
        panel_height = objects_per_column * (bar_height + bar_spacing) + padding * 2
        if horizontal == "right":
            panel_x = img_width - panel_width - padding
        elif horizontal == "center":
            panel_x = (img_width - panel_width) // 2
        else:
            panel_x = padding

    # Vertical placement
    if vertical == "top":
        panel_y = padding
    elif vertical == "center":
        panel_y = (img_height - panel_height) // 2
    else:
        panel_y = img_height - panel_height - padding

    # ── Frame-by-frame rendering ──────────────────────────────────────
    processed_imgs = []
    for frame_idx in range(T):
        if frame_idx >= len(imgs):
            break

        img = imgs[frame_idx].copy()
        if img.dtype != np.uint8:
            img = (
                (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            )

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Semi-transparent background panel
        overlay = img_bgr.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            _COLORS["bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

        for obj_idx, obj_name in enumerate(target_objects):
            h_arr = health.get(obj_name)
            if h_arr is None:
                current_health = 100.0
            elif frame_idx < len(h_arr):
                current_health = float(h_arr[frame_idx])
            else:
                current_health = 100.0
            current_health = max(0.0, min(100.0, current_health))

            if use_row_layout:
                item_x = panel_x + padding + obj_idx * (item_width + row_spacing)
                bar_y = panel_y + padding
                label_x = item_x
                bar_x_start = label_x + label_width + gap_after_label
                value_x = bar_x_start + bar_width + gap_after_bar
            else:
                col_idx = obj_idx // objects_per_column
                row_idx = obj_idx % objects_per_column
                column_x = panel_x + col_idx * (column_width + column_spacing)
                bar_y = panel_y + padding + row_idx * (bar_height + bar_spacing)
                label_x = column_x + padding
                bar_x_start = label_x + label_width + gap_after_label
                value_x = bar_x_start + bar_width + gap_after_bar

            # (label_x, bar_x_start, value_x, bar_y set above for both layouts)

            # Background bar container
            cv2.rectangle(
                img_bgr,
                (bar_x_start, bar_y),
                (bar_x_start + bar_width, bar_y + bar_height),
                _COLORS["bg_bar"],
                -1,
            )
            cv2.rectangle(
                img_bgr,
                (bar_x_start, bar_y),
                (bar_x_start + bar_width, bar_y + bar_height),
                _COLORS["border"],
                2,
            )

            # Glow at bottom edge
            glow_height = 3
            cv2.rectangle(
                img_bgr,
                (bar_x_start, bar_y + bar_height - glow_height),
                (bar_x_start + bar_width, bar_y + bar_height),
                _COLORS["glow"],
                -1,
            )

            # Health bar colour
            health_width = int((current_health / 100.0) * (bar_width - 6))
            bar_inset = 3

            if current_health == 0:
                bar_color = None
                value_color = _COLORS["text_dark_gray"]
            elif current_health >= 80:
                bar_color = _COLORS["green"]
                value_color = _COLORS["text_white"]
            elif current_health >= 60:
                bar_color = _COLORS["amber"]
                value_color = _COLORS["text_white"]
            elif current_health >= 40:
                bar_color = _COLORS["orange"]
                value_color = _COLORS["text_white"]
            elif current_health >= 20:
                bar_color = _COLORS["red"]
                value_color = _COLORS["text_white"]
            else:
                bar_color = _COLORS["dark_red"]
                value_color = _COLORS["text_white"]

            # Foreground bar
            if bar_color is not None and health_width > 0:
                cv2.rectangle(
                    img_bgr,
                    (bar_x_start + bar_inset, bar_y + bar_inset),
                    (bar_x_start + bar_inset + health_width, bar_y + bar_height - bar_inset),
                    bar_color,
                    -1,
                )

            # Label
            display_name = display_map.get(obj_name, obj_name)
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            label_color = (
                _COLORS["text_gray"] if current_health == 0 else _COLORS["text_light"]
            )
            cv2.putText(
                img_bgr,
                display_name,
                (label_x, bar_y + bar_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                label_color,
                1,
                cv2.LINE_AA,
            )

            # Health value text
            health_text = f"{current_health:.1f}"
            cv2.putText(
                img_bgr,
                health_text,
                (value_x, bar_y + bar_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                value_color,
                1,
                cv2.LINE_AA,
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed_imgs.append(img_rgb)

    # Delegate to camera video saver (AVI → MP4)
    save_rgb_camera_video(output_video_path, np.array(processed_imgs), fps=fps)

def save_rgb_health_video(
    output_video_path: str,
    imgs: np.ndarray,
    target_objects: List[str],
    health: Dict[str, np.ndarray],
    fps: int = 30,
) -> None:
    """
    Save a side-by-side video: RGB on the left, health line plot on the right.

    Args:
        output_video_path: Destination path (with or without ``.mp4``).
        imgs: ``(T, H, W, 3)`` uint8 array in **RGB** order.
        target_objects: Object names whose health to plot.
        health: Mapping ``obj_name -> 1-D array of health values (0-100)``.
        fps: Frames per second.
    """
    T = len(health[target_objects[0]])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    ax_health = fig.add_subplot(gs[0, 1])
    ax_health.set_title("Health Over Time")
    ax_health.set_xlabel("Time (s)")
    ax_health.set_ylabel("Health")
    ax_health.set_xlim(0, T / fps)
    ax_health.set_ylim(-5.0, 105.0)
    ax_health.grid(True)

    health_lines = {}
    for obj_name in target_objects:
        health_lines[obj_name], = ax_health.plot(
            [], [], lw=2, label=f"{obj_name} Health",
        )

    ax_health.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    time = [i / fps for i in range(T)]

    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in health_lines.values():
            line.set_data([], [])
        return [video_im] + list(health_lines.values())

    def animate(i):
        video_im.set_data(imgs[i][:, :, :3])
        for obj_name in target_objects:
            health_lines[obj_name].set_data(
                time[: i + 1], health[obj_name][: i + 1],
            )
        return [video_im] + list(health_lines.values())

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=T, interval=1000 / fps, blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)


def save_rgb_force_video(
    output_video_path,
    imgs,
    target_objects,
    data,
    forces_to_plot=("dynamic_forces", "static_forces", "raw_forces_from_sim"),
    fps=30,
):
    T = len(data[target_objects[0]][forces_to_plot[0]])

    # ---------------------------
    # FIGURE: 1 row, 2 columns
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ---------------------------
    # LEFT: RGB VIDEO
    # ---------------------------
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(imgs[0][:, :, :3])

    # ---------------------------
    # RIGHT: FORCE PLOT
    # ---------------------------
    ax_force = fig.add_subplot(gs[0, 1])
    ax_force.set_title("Force History")
    ax_force.set_xlabel("Time (s)")
    ax_force.set_ylabel("End-effector Force (N)")
    ax_force.set_xlim(0, T / fps)
    ax_force.set_ylim(0, 500.0)
    ax_force.grid(True)

    force_lines = {}

    for obj_name in target_objects:
        for force_key in forces_to_plot:
            force_lines[f"{obj_name}_{force_key}"], = ax_force.plot(
                [],
                [],
                lw=2,
                label=f"{obj_name} {force_key}",
            )

    ax_force.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.25)

    # Precompute time axis
    time = [i / fps for i in range(T)]

    # ---------------------------
    # INIT
    # ---------------------------
    def init():
        video_im.set_data(imgs[0][:, :, :3])
        for line in force_lines.values():
            line.set_data([], [])
        return [video_im] + list(force_lines.values())

    # ---------------------------
    # ANIMATE
    # ---------------------------
    def animate(i):
        # RGB frame
        video_im.set_data(imgs[i][:, :, :3])

        # Force plot
        for obj_name in target_objects:
            for force_key in forces_to_plot:
                force_lines[f"{obj_name}_{force_key}"].set_data(
                    time[: i + 1],
                    data[obj_name][force_key][: i + 1],
                )

        return [video_im] + list(force_lines.values())

    # ---------------------------
    # SAVE
    # ---------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            "-hide_banner",
            "-nostats",
        ],
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Colour palette used by the health-bar overlay
_COLORS = {
    "bg": _hex_to_bgr("#1A1A1A"),
    "bg_bar": _hex_to_bgr("#0D0D0D"),
    "border": _hex_to_bgr("#2A2A2A"),
    "glow": _hex_to_bgr("#333333"),
    "green": _hex_to_bgr("#4CAF50"),
    "amber": _hex_to_bgr("#FFC107"),
    "orange": _hex_to_bgr("#FF9800"),
    "red": _hex_to_bgr("#F44336"),
    "dark_red": _hex_to_bgr("#D32F2F"),
    "text_white": _hex_to_bgr("#FFFFFF"),
    "text_light": _hex_to_bgr("#E8E8E8"),
    "text_gray": _hex_to_bgr("#888888"),
    "text_dark_gray": _hex_to_bgr("#666666"),
}

def render_health_bar_overlay(
    frame_rgb: np.ndarray,
    target_objects: List[str],
    current_health: Dict[str, float],
    position: str = "bottom_left",
    n_columns: int = 1,
    obj_display_names: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Render health bar overlay on a single RGB frame.

    Dimensions and font sizes scale automatically with the frame resolution,
    so the same call works for both real-time HUD (e.g. 1280x720) and
    high-resolution screenshots (e.g. 3840x2160).

    Args:
        frame_rgb: ``(H, W, 3)`` uint8 array in **RGB** order.
        target_objects: Object names to show health bars for.
        current_health: Mapping ``obj_name -> health`` (0–100).
        position: ``"bottom_left"``, ``"bottom_right"``, or
            ``"bottom_center"`` / ``"center"``.
        n_columns: Number of columns for laying out health bars (1–3).
        obj_display_names: Optional override for display names.

    Returns:
        ``(H, W, 3)`` uint8 array in **RGB** order with health bars overlaid.
    """
    if not target_objects:
        return frame_rgb

    display_map = dict(OBJ_NAME_DISPLAY_NAME_MAPPING)
    if obj_display_names:
        display_map.update(obj_display_names)

    n_columns = max(1, min(3, int(n_columns)))
    n_objects = len(target_objects)

    img_height, img_width = frame_rgb.shape[:2]

    # ── Bar dimensions (scale with image size) ────────────────────────
    bar_height = max(25, int(img_height * 0.05))
    bar_spacing = max(8, int(img_height * 0.01))
    padding_h = max(20, int(img_width * 0.025))
    padding_v = max(8, int(img_height * 0.012))
    column_spacing = max(15, int(img_width * 0.02))

    label_width = max(80, int(img_width * 0.13))
    bar_width = max(150, int(img_width * 0.18))
    gap_after_label = max(4, int(img_width * 0.012))
    gap_after_bar = max(8, int(img_width * 0.012))
    font_size = max(0.4, min(1.5, img_width / 1600.0))

    objects_per_column = int(np.ceil(n_objects / n_columns))
    value_width = max(50, int(img_width * 0.06))
    column_width = (
        label_width
        + gap_after_label
        + bar_width
        + gap_after_bar
        + value_width
    )
    panel_width = n_columns * column_width + (n_columns - 1) * column_spacing + padding_h * 2
    panel_height = objects_per_column * (bar_height + bar_spacing) + padding_v * 2

    # ── Panel position ────────────────────────────────────────────────
    if position == "bottom_right":
        panel_x = img_width - panel_width - padding_h
    elif position in ("center", "bottom_center"):
        panel_x = (img_width - panel_width) // 2
    else:
        panel_x = padding_h
    panel_y = img_height - panel_height - padding_v

    # ── Render ────────────────────────────────────────────────────────
    img = frame_rgb.copy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Semi-transparent background panel
    overlay = img_bgr.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        _COLORS["bg"],
        -1,
    )
    cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

    for obj_idx, obj_name in enumerate(target_objects):
        h = current_health.get(obj_name, 100.0)
        h = max(0.0, min(100.0, h))

        col_idx = obj_idx // objects_per_column
        row_idx = obj_idx % objects_per_column

        column_x = panel_x + col_idx * (column_width + column_spacing)
        bar_y = panel_y + padding_v + row_idx * (bar_height + bar_spacing)
        label_x = column_x + padding_h
        bar_x_start = label_x + label_width + gap_after_label
        value_x = bar_x_start + bar_width + gap_after_bar

        # Background bar container
        cv2.rectangle(
            img_bgr,
            (bar_x_start, bar_y),
            (bar_x_start + bar_width, bar_y + bar_height),
            _COLORS["bg_bar"],
            -1,
        )
        cv2.rectangle(
            img_bgr,
            (bar_x_start, bar_y),
            (bar_x_start + bar_width, bar_y + bar_height),
            _COLORS["border"],
            2,
        )

        # Glow at bottom edge
        glow_height = 3
        cv2.rectangle(
            img_bgr,
            (bar_x_start, bar_y + bar_height - glow_height),
            (bar_x_start + bar_width, bar_y + bar_height),
            _COLORS["glow"],
            -1,
        )

        # Health bar colour
        health_width = int((h / 100.0) * (bar_width - 6))
        bar_inset = 3

        if h == 0:
            bar_color = None
            value_color = _COLORS["text_dark_gray"]
        elif h >= 80:
            bar_color = _COLORS["green"]
            value_color = _COLORS["text_white"]
        elif h >= 60:
            bar_color = _COLORS["amber"]
            value_color = _COLORS["text_white"]
        elif h >= 40:
            bar_color = _COLORS["orange"]
            value_color = _COLORS["text_white"]
        elif h >= 20:
            bar_color = _COLORS["red"]
            value_color = _COLORS["text_white"]
        else:
            bar_color = _COLORS["dark_red"]
            value_color = _COLORS["text_white"]

        # Foreground bar
        if bar_color is not None and health_width > 0:
            cv2.rectangle(
                img_bgr,
                (bar_x_start + bar_inset, bar_y + bar_inset),
                (bar_x_start + bar_inset + health_width, bar_y + bar_height - bar_inset),
                bar_color,
                -1,
            )

        # Label
        display_name = display_map.get(obj_name, obj_name)
        if len(display_name) > 20:
            display_name = display_name[:17] + "..."
        label_color = (
            _COLORS["text_gray"] if h == 0 else _COLORS["text_light"]
        )
        cv2.putText(
            img_bgr,
            display_name,
            (label_x, bar_y + bar_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            label_color,
            1,
            cv2.LINE_AA,
        )

        # Health value text
        health_text = f"{h:.1f}"
        cv2.putText(
            img_bgr,
            health_text,
            (value_x, bar_y + bar_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            value_color,
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)