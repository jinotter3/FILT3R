import argparse
import shutil
from pathlib import Path

import numpy as np


DEFAULT_TARGET_FRAMES = [
    50,
    90,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
    1000,
]


def parse_frame_list(value):
    return [int(item) for item in value.replace(",", " ").split()]


def frame_sort_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def load_valid_triplets(scene_dir, sample_interval):
    img_paths = sorted((scene_dir / "color").glob("*.jpg"), key=frame_sort_key)
    depth_paths = sorted((scene_dir / "depth").glob("*.png"), key=frame_sort_key)
    pose_paths = sorted((scene_dir / "pose").glob("*.txt"), key=frame_sort_key)
    total_frames = min(len(img_paths), len(depth_paths), len(pose_paths))

    sampled_triplets = []
    skipped_non_finite = 0
    for idx in range(0, total_frames, sample_interval):
        pose = np.loadtxt(pose_paths[idx]).reshape(-1)
        if not np.isfinite(pose).all():
            skipped_non_finite += 1
            continue
        sampled_triplets.append((img_paths[idx], depth_paths[idx], pose))

    return total_frames, sampled_triplets, skipped_non_finite


def copy_scene(scene_dir, output_root, frame_budget, sample_interval, overwrite):
    total_frames, triplets, skipped_non_finite = load_valid_triplets(scene_dir, sample_interval)
    selected_triplets = triplets[: min(frame_budget, len(triplets))]

    scene_out = output_root / scene_dir.name
    color_out = scene_out / f"color_{frame_budget}"
    depth_out = scene_out / f"depth_{frame_budget}"
    pose_out = scene_out / f"pose_{frame_budget}.txt"

    if overwrite:
        shutil.rmtree(color_out, ignore_errors=True)
        shutil.rmtree(depth_out, ignore_errors=True)
        if pose_out.exists():
            pose_out.unlink()
    color_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    for index, (img_path, depth_path, _) in enumerate(selected_triplets):
        shutil.copy2(img_path, color_out / f"frame_{index:04d}.jpg")
        shutil.copy2(depth_path, depth_out / f"frame_{index:04d}.png")

    with pose_out.open("w", encoding="utf-8") as handle:
        for _, _, pose in selected_triplets:
            handle.write(f"{' '.join(map(str, pose))}\n")

    print(
        f"[scannet] {scene_dir.name}: original={total_frames} target={frame_budget} "
        f"frames={len(selected_triplets)} skipped_non_finite={skipped_non_finite}"
    )


def prepare_scannet(input_root, output_root, target_frames, sample_interval, overwrite):
    scene_dirs = sorted(
        path for path in input_root.iterdir() if path.is_dir() and not path.name.endswith(".sens")
    )
    if not scene_dirs:
        raise FileNotFoundError(f"No ScanNet scene directories found under {input_root}.")

    for frame_budget in target_frames:
        for scene_dir in scene_dirs:
            copy_scene(
                scene_dir=scene_dir,
                output_root=output_root,
                frame_budget=frame_budget,
                sample_interval=sample_interval,
                overwrite=overwrite,
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare long-horizon ScanNet folders for FILT3R evaluation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="ScanNet v2 root containing scene/color, scene/depth, and scene/pose folders.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/long_scannet_s3"))
    parser.add_argument("--sample-interval", type=int, default=3)
    parser.add_argument(
        "--target-frames",
        type=parse_frame_list,
        default=DEFAULT_TARGET_FRAMES,
        help="Comma- or space-separated frame budgets.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing output folders instead of recreating them.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    prepare_scannet(
        input_root=args.input_root.expanduser().resolve(),
        output_root=args.output_root.expanduser(),
        target_frames=args.target_frames,
        sample_interval=args.sample_interval,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
