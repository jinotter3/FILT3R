import argparse
import shutil
from pathlib import Path

import numpy as np


DEFAULT_TARGET_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def parse_frame_list(value):
    return [int(item) for item in value.replace(",", " ").split()]


def load_groundtruth(path):
    gt = np.loadtxt(path)
    return np.atleast_2d(gt)


def copy_sequence(scene_dir, output_root, start_frame, frame_budget, overwrite):
    rgb_frames = sorted((scene_dir / "rgb").glob("*.png"))
    depth_frames = sorted((scene_dir / "depth").glob("*.png"))
    gt_path = scene_dir / "groundtruth.txt"
    if not gt_path.exists():
        print(f"[bonn] skipping {scene_dir.name}: missing groundtruth.txt")
        return

    gt = load_groundtruth(gt_path)
    available = min(len(rgb_frames), len(depth_frames), len(gt))
    final_count = max(0, min(available - start_frame, frame_budget))
    if final_count == 0:
        print(f"[bonn] skipping {scene_dir.name}: no frames in requested range")
        return

    scene_out = output_root / scene_dir.name
    rgb_out = scene_out / f"rgb_{frame_budget}"
    depth_out = scene_out / f"depth_{frame_budget}"
    gt_out = scene_out / f"groundtruth_{frame_budget}.txt"

    if overwrite:
        shutil.rmtree(rgb_out, ignore_errors=True)
        shutil.rmtree(depth_out, ignore_errors=True)
        if gt_out.exists():
            gt_out.unlink()
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    rgb_slice = rgb_frames[start_frame : start_frame + final_count]
    depth_slice = depth_frames[start_frame : start_frame + final_count]
    for frame in rgb_slice:
        shutil.copy2(frame, rgb_out / frame.name)
    for frame in depth_slice:
        shutil.copy2(frame, depth_out / frame.name)

    np.savetxt(gt_out, gt[start_frame : start_frame + final_count])
    print(f"[bonn] {scene_dir.name}: target={frame_budget} frames={final_count}")


def prepare_bonn(input_root, output_root, target_frames, start_frame, overwrite):
    scene_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No Bonn sequence directories found under {input_root}.")

    for frame_budget in target_frames:
        for scene_dir in scene_dirs:
            copy_sequence(
                scene_dir=scene_dir,
                output_root=output_root,
                start_frame=start_frame,
                frame_budget=frame_budget,
                overwrite=overwrite,
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare long-horizon Bonn RGB-D folders for FILT3R evaluation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Bonn rgbd_bonn_dataset root containing per-sequence rgb/depth/groundtruth files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/long_bonn_s1/rgbd_bonn_dataset"),
    )
    parser.add_argument("--start-frame", type=int, default=30)
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
    prepare_bonn(
        input_root=args.input_root.expanduser().resolve(),
        output_root=args.output_root.expanduser(),
        target_frames=args.target_frames,
        start_frame=args.start_frame,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
