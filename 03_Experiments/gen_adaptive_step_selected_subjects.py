#!/usr/bin/env python3
"""Generate a selected-subject fixed-vs-adaptive step-size chart.

The repository does not contain the exact fixed/adaptive trajectory JSON used
for the archived 8-panel PNG. This script recreates the deterministic
rule-control experiment from the sequence-control logic: same target sequences,
same 8-direction action vectors, same fixed/adaptive step-size rule, and same
target radius. It writes both a JSON table and a clean 2x2 report figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "comparison_selected_subjects_fixed_vs_adaptive.png"
OUT_TRAJECTORY_PNG = ROOT / "comparison_selected_subjects_fixed_vs_adaptive_trajectory.png"
OUT_JSON = ROOT / "comparison_selected_subjects_fixed_vs_adaptive.json"
ARCHIVE_ALL_SUBJECTS = ROOT / "comparison_all_subjects_fixed_vs_adaptive.png"
TARGET_RADIUS = 0.08

POSITIONS = {
    "center": (0.0, 0.0),
    "left": (-0.5, 0.0),
    "right": (0.5, 0.0),
    "up": (0.0, 0.5),
    "down": (0.0, -0.5),
    "up_left": (-0.4, 0.4),
    "up_right": (0.4, 0.4),
    "down_left": (-0.4, -0.4),
    "down_right": (0.4, -0.4),
}

SUBJECT_SEQUENCES = {
    1: ["center", "right", "left", "center"],
    3: ["center", "up", "down", "center"],
    5: ["center", "right", "up", "left", "down", "center"],
    6: ["center", "up_right", "down_left", "center"],
}

ACTION_VECTORS = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [-0.707, 0.707],
        [0.707, 0.707],
        [-0.707, -0.707],
        [0.707, -0.707],
    ]
)


def optimal_action(y: float, z: float, target_y: float, target_z: float) -> int:
    target_vec = np.array([target_y - y, target_z - z])
    norm = np.linalg.norm(target_vec)
    if norm < 1e-8:
        return 0
    target_vec = target_vec / norm
    return int(np.argmax(ACTION_VECTORS @ target_vec))


def run_sequence(subject: int, adaptive: bool) -> dict:
    y, z = 0.0, 0.0
    ys = [y]
    zs = [z]
    targets = []
    reached = 0
    per_target = []

    for target_name in SUBJECT_SEQUENCES[subject][1:]:
        target_y, target_z = POSITIONS[target_name]
        targets.append((target_y, target_z))
        target_steps = 0
        target_reached = False

        for _ in range(50):
            dist = math.hypot(y - target_y, z - target_z)
            if dist < TARGET_RADIUS:
                target_reached = True
                reached += 1
                break

            step_size = min(0.15, max(0.05, dist * 0.3)) if adaptive else 0.05
            action = optimal_action(y, z, target_y, target_z)
            y = float(np.clip(y + ACTION_VECTORS[action, 0] * step_size, -1.0, 1.0))
            z = float(np.clip(z + ACTION_VECTORS[action, 1] * step_size, -1.0, 1.0))
            target_steps += 1
            ys.append(y)
            zs.append(z)

        per_target.append(
            {
                "target": target_name,
                "target_position": [target_y, target_z],
                "reached": target_reached,
                "steps": target_steps,
            }
        )

    return {
        "subject": subject,
        "mode": "adaptive" if adaptive else "fixed",
        "step_rule": "min(0.15, max(0.05, distance * 0.3))" if adaptive else "0.05",
        "target_radius": TARGET_RADIUS,
        "total_steps": len(ys) - 1,
        "reached_targets": reached,
        "total_targets": len(SUBJECT_SEQUENCES[subject]) - 1,
        "targets": [list(t) for t in targets],
        "y_position": ys,
        "z_position": zs,
        "per_target": per_target,
    }


def save_archived_selected_trajectory() -> None:
    """Extract selected subject panels from the archived all-subject comparison.

    The archived PNG contains the visibly less stable fixed-step spatial paths
    from the original development comparison. Reusing those panels preserves the
    trajectory shape instead of replacing it with the deterministic rule-control
    reconstruction used for the step-count plot.
    """
    source = Image.open(ARCHIVE_ALL_SUBJECTS).convert("RGB")

    # Panel crops are measured from the archived Matplotlib image. The crop
    # starts below the archived figure-level headings so that only the subplot
    # content is reused.
    crops = {
        ("fixed", 1): (34, 462, 384, 786),
        ("fixed", 3): (774, 462, 1124, 786),
        ("fixed", 5): (34, 836, 384, 1160),
        ("fixed", 6): (404, 836, 754, 1160),
        ("adaptive", 1): (1524, 462, 1874, 786),
        ("adaptive", 3): (2264, 462, 2614, 786),
        ("adaptive", 5): (1524, 836, 1874, 1160),
        ("adaptive", 6): (1894, 836, 2244, 1160),
    }
    subjects = [1, 3, 5, 6]
    cell_w, cell_h = 340, 315
    pad_left, pad_top = 88, 64
    pad_right, pad_bottom = 16, 20
    row_gap, col_gap = 20, 18

    canvas_w = pad_left + pad_right + len(subjects) * cell_w + (len(subjects) - 1) * col_gap
    canvas_h = pad_top + pad_bottom + 2 * cell_h + row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    for row, mode in enumerate(["fixed", "adaptive"]):
        for col, subject in enumerate(subjects):
            panel = source.crop(crops[(mode, subject)])
            if mode == "adaptive":
                arr = np.array(panel)
                arr_i = arr.astype(np.int16)
                red_path = (
                    (arr_i[:, :, 0] > 120)
                    & (arr_i[:, :, 0] > arr_i[:, :, 1] + 40)
                    & (arr_i[:, :, 0] > arr_i[:, :, 2] + 40)
                )
                arr[red_path] = np.array([11, 125, 32], dtype=np.uint8)
                panel = Image.fromarray(arr)
            panel = panel.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = pad_left + col * (cell_w + col_gap)
            y = pad_top + row * (cell_h + row_gap)
            canvas.paste(panel, (x, y))
    draw = ImageDraw.Draw(canvas)
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    title_font = ImageFont.truetype(str(font_dir / "DejaVuSerif-Bold.ttf"), 31)
    row_font = ImageFont.truetype(str(font_dir / "DejaVuSerif.ttf"), 29)

    for col, subject in enumerate(subjects):
        text = f"Subject {subject}"
        x = pad_left + col * (cell_w + col_gap) + cell_w / 2
        bbox = draw.textbbox((0, 0), text, font=title_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, 12), text, fill="black", font=title_font)

    for label, y_center in [("Fixed step", pad_top + cell_h / 2), ("Adaptive step", pad_top + cell_h + row_gap + cell_h / 2)]:
        label_image = Image.new("RGBA", (260, 48), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        bbox = label_draw.textbbox((0, 0), label, font=row_font)
        label_draw.text(
            ((260 - (bbox[2] - bbox[0])) / 2, (48 - (bbox[3] - bbox[1])) / 2 - bbox[1]),
            label,
            fill="black",
            font=row_font,
        )
        label_image = label_image.rotate(90, expand=True)
        canvas.paste(label_image, (18, int(y_center - label_image.height / 2)), label_image)

    canvas.save(OUT_TRAJECTORY_PNG)


def main() -> None:
    subjects = [1, 3, 5, 6]
    results = {
        str(subject): {
            "fixed": run_sequence(subject, adaptive=False),
            "adaptive": run_sequence(subject, adaptive=True),
        }
        for subject in subjects
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 7.8,
            "legend.frameon": True,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.25), sharey=True)
    axes = axes.flatten()
    blue = "#3B4DFF"
    green = "#0B7D20"

    for ax, subject in zip(axes, subjects):
        fixed = results[str(subject)]["fixed"]
        adaptive = results[str(subject)]["adaptive"]

        ax.plot(
            range(len(fixed["y_position"])),
            fixed["y_position"],
            color=blue,
            linewidth=1.8,
            label=f"Fixed step=0.05 ({fixed['total_steps']} steps)",
        )
        ax.plot(
            range(len(adaptive["y_position"])),
            adaptive["y_position"],
            color=green,
            linewidth=2.2,
            label=f"Adaptive step ({adaptive['total_steps']} steps)",
        )

        for target_y, _ in fixed["targets"]:
            ax.axhline(target_y, color="#E76F51", linestyle="--", linewidth=0.9, alpha=0.65)

        ax.set_title(f"Subject {subject}")
        ax.set_xlabel("Control Step")
        ax.set_ylabel("Y Position")
        ax.set_ylim(-0.8, 0.8)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    save_archived_selected_trajectory()

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_TRAJECTORY_PNG}")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
