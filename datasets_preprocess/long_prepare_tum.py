import argparse
import shutil
from pathlib import Path


DEFAULT_TARGET_FRAMES = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def parse_frame_list(value):
    return [int(item) for item in value.replace(",", " ").split()]


def read_file_list(filename):
    data = Path(filename).read_text(encoding="utf-8")
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    entries = [
        [value.strip() for value in line.split(" ") if value.strip()]
        for line in lines
        if line and not line.startswith("#")
    ]
    return {float(entry[0]): entry[1:] for entry in entries if len(entry) > 1}


def associate(first_list, second_list, offset, max_difference):
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()

    matches = []
    for _, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def load_matched_frames(sequence_dir, max_difference):
    rgb_file = sequence_dir / "rgb.txt"
    gt_file = sequence_dir / "groundtruth.txt"
    if not rgb_file.exists() or not gt_file.exists():
        print(f"[tum] skipping {sequence_dir.name}: missing rgb.txt or groundtruth.txt")
        return [], []

    rgb_list = read_file_list(rgb_file)
    gt_list = read_file_list(gt_file)
    matches = associate(rgb_list, gt_list, offset=0.0, max_difference=max_difference)

    frames = []
    poses = []
    for rgb_stamp, gt_stamp in matches:
        frames.append(sequence_dir / rgb_list[rgb_stamp][0])
        poses.append([gt_stamp] + gt_list[gt_stamp])
    return frames, poses


def copy_sequence(sequence_dir, output_root, frame_budget, sample_interval, max_difference, overwrite):
    frames, poses = load_matched_frames(sequence_dir, max_difference)
    if not frames:
        return

    selected_frames = frames[::sample_interval][:frame_budget]
    selected_poses = poses[::sample_interval][:frame_budget]
    if not selected_frames:
        print(f"[tum] skipping {sequence_dir.name}: no frames after sampling")
        return

    sequence_out = output_root / sequence_dir.name
    rgb_out = sequence_out / f"rgb_{frame_budget}"
    gt_out = sequence_out / f"groundtruth_{frame_budget}.txt"

    if overwrite:
        shutil.rmtree(rgb_out, ignore_errors=True)
        if gt_out.exists():
            gt_out.unlink()
    rgb_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for frame in selected_frames:
        if frame.exists():
            shutil.copy2(frame, rgb_out / frame.name)
            copied += 1
        else:
            print(f"[tum] missing RGB frame: {frame}")

    with gt_out.open("w", encoding="utf-8") as handle:
        for pose in selected_poses:
            handle.write(f"{' '.join(map(str, pose))}\n")

    print(f"[tum] {sequence_dir.name}: target={frame_budget} frames={copied}")


def prepare_tum(input_root, output_root, target_frames, sample_interval, max_difference, overwrite):
    sequence_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError(f"No TUM sequence directories found under {input_root}.")

    for frame_budget in target_frames:
        for sequence_dir in sequence_dirs:
            copy_sequence(
                sequence_dir=sequence_dir,
                output_root=output_root,
                frame_budget=frame_budget,
                sample_interval=sample_interval,
                max_difference=max_difference,
                overwrite=overwrite,
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare long-horizon TUM RGB/groundtruth folders for FILT3R evaluation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="TUM root containing per-sequence rgb.txt and groundtruth.txt files.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/long_tum_s1"))
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--max-difference", type=float, default=0.02)
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
    prepare_tum(
        input_root=args.input_root.expanduser().resolve(),
        output_root=args.output_root.expanduser(),
        target_frames=args.target_frames,
        sample_interval=args.sample_interval,
        max_difference=args.max_difference,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
