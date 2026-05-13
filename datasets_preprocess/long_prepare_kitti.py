import argparse
import shutil
from pathlib import Path

from PIL import Image
import numpy as np


DEFAULT_TARGET_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def parse_frame_list(value):
    return [int(item) for item in value.replace(",", " ").split()]


def depth_read(filename):
    depth_png = np.array(Image.open(filename), dtype=np.int32)
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def resolve_raw_image(depth_file, val_root, raw_root):
    rel = depth_file.relative_to(val_root)
    sequence = rel.parts[0]
    date = "_".join(sequence.split("_")[:3])
    frame_name = rel.name
    return raw_root / date / sequence / "image_02" / "data" / frame_name


def prepare_kitti(input_root, output_root, raw_root, target_frames, overwrite):
    depth_dirs = sorted(
        input_root.glob("*/proj_depth/groundtruth/image_02"),
        key=lambda path: path.parent.parent.parent.name,
    )
    if not depth_dirs:
        raise FileNotFoundError(
            f"No KITTI depth folders found under {input_root}. Expected */proj_depth/groundtruth/image_02."
        )

    for frame_budget in target_frames:
        for depth_dir in depth_dirs:
            sequence = depth_dir.parent.parent.parent.name
            seq_name = f"{sequence}_02"
            new_depth_dir = output_root / f"groundtruth_depth_gathered_{frame_budget}" / seq_name
            new_image_dir = output_root / f"image_gathered_{frame_budget}" / seq_name

            if overwrite:
                shutil.rmtree(new_depth_dir, ignore_errors=True)
                shutil.rmtree(new_image_dir, ignore_errors=True)
            new_depth_dir.mkdir(parents=True, exist_ok=True)
            new_image_dir.mkdir(parents=True, exist_ok=True)

            depth_files = sorted(depth_dir.glob("*.png"))[:frame_budget]
            copied_images = 0
            for depth_file in depth_files:
                shutil.copy2(depth_file, new_depth_dir / depth_file.name)
                image_file = resolve_raw_image(depth_file, input_root, raw_root)
                if image_file.exists():
                    shutil.copy2(image_file, new_image_dir / image_file.name)
                    copied_images += 1
                else:
                    print(f"[kitti] missing RGB image for depth {depth_file}: {image_file}")

            print(
                f"[kitti] {sequence}: target={frame_budget} "
                f"depth={len(depth_files)} rgb={copied_images}"
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare long-horizon KITTI depth/image folders for FILT3R evaluation."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="KITTI validation root containing sequence/proj_depth folders, e.g. /path/to/kitti/val.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="KITTI raw root containing date/sequence/image_02/data. Defaults to the parent of --input-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/long_kitti_s1/depth_selection/val_selection_cropped"),
    )
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
    input_root = args.input_root.expanduser().resolve()
    raw_root = (args.raw_root or input_root.parent).expanduser().resolve()
    output_root = args.output_root.expanduser()
    prepare_kitti(
        input_root=input_root,
        output_root=output_root,
        raw_root=raw_root,
        target_frames=args.target_frames,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
