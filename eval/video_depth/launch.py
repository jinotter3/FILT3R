import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import ast
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
import time
from tqdm import tqdm


def _parse_hparam_value(raw_value: str):
    value_str = raw_value.strip()
    low = value_str.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


FILT3R_DEFAULT_HPARAMS = {
    "kalman_p_init": 1.5,
    "kalman_gamma_p": 1.0,
    "kalman_q_min": 0.02,
    "kalman_q_max": 0.5,
    "kalman_alpha_q": 20.0,
    "kalman_tau_q": 3.0,
    "kalman_ema_beta_delta": 0.05,
    "kalman_ema_delta_floor": 1e-2,
    "kalman_fixed_r": 1.0,
}


VIDEO_DEPTH_SEQ_DONE_FILENAME = "_seq_done.txt"


def _video_depth_done_marker_path(save_dir, seq):
    return os.path.join(save_dir, seq, VIDEO_DEPTH_SEQ_DONE_FILENAME)


def _count_saved_depth_frames(seq_dir):
    if not os.path.isdir(seq_dir):
        return 0
    return sum(
        1
        for name in os.listdir(seq_dir)
        if name.startswith("frame_") and name.endswith(".npy")
    )


def _is_video_depth_sequence_done(save_dir, seq, expected_frames):
    # New-style explicit completion marker.
    done_marker_path = _video_depth_done_marker_path(save_dir, seq)
    if os.path.exists(done_marker_path):
        return True

    # Backward-compatible check for legacy outputs from older runs.
    if expected_frames <= 0:
        return False
    seq_dir = os.path.join(save_dir, seq)
    return _count_saved_depth_frames(seq_dir) == expected_frames


def _mark_video_depth_sequence_done(save_dir, seq):
    marker_path = _video_depth_done_marker_path(save_dir, seq)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        f.write("done\n")


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        choices=[
            "cut3r",
            "ttt3r",
            "filt3r",
        ],
        help="state update strategy",
    )
    parser.add_argument(
        "--model_hparam",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="generic model hparam override (repeatable), e.g. --model_hparam kalman_p_init=1.5",
    )


    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="skip sequences that already have completed outputs",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="recompute all sequences even if outputs already exist",
    )
    parser.set_defaults(resume=True)
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    distributed_state = PartialState()
    model.to(distributed_state.device)
    if distributed_state.device.type == "cpu":
        model.float()
    device = distributed_state.device
    if distributed_state.is_main_process:
        print(f"Resume mode: {'ON' if args.resume else 'OFF'}")

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                if args.resume and _is_video_depth_sequence_done(
                    save_dir, seq, len(filelist)
                ):
                    print(f"Skipping completed sequence: {args.eval_dataset} {seq}")
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"{args.eval_dataset}-{seq: <16} | skipped (already complete)\n"
                        )
                    continue

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                )
                start = time.time()
                outputs, _ = inference_recurrent_lighter(views, model, device)
                end = time.time()
                fps = len(filelist) / (end - start)

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(outputs)

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_depth_maps(pts3ds_self, f"{save_dir}/{seq}", conf_self=conf_self)
                _mark_video_depth_sequence_done(save_dir, seq)

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import matrix_cumprod

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0).to(
            dtype=torch.bool
        )
        shifted_reset_mask = torch.cat(
            [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
        )
        outputs["pred"] = [
            pred
            for pred, mask in zip(outputs["pred"], shifted_reset_mask.tolist())
            if not mask
        ]
        outputs["views"] = [
            view
            for view, mask in zip(outputs["views"], shifted_reset_mask.tolist())
            if not mask
        ]
        reset_mask = reset_mask[~shifted_reset_mask]

        pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)
        if reset_mask.any():
            identity = torch.eye(4, device=pr_poses.device, dtype=pr_poses.dtype)
            reset_poses = torch.where(
                reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity
            )
            cumulative_bases = matrix_cumprod(reset_poses)
            shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
            pr_poses = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses)

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = ARCroco3DStereo.from_pretrained(args.weights)

    # set model type
    model.config.model_update_type = args.model_update_type
    hparam_overrides = {}
    if args.model_update_type == "filt3r":
        hparam_overrides.update(FILT3R_DEFAULT_HPARAMS)
    for item in args.model_hparam:
        if "=" not in item:
            raise ValueError(
                f"Invalid --model_hparam '{item}'. Expected KEY=VALUE format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key == "":
            raise ValueError(
                f"Invalid --model_hparam '{item}'. Key cannot be empty."
            )
        hparam_overrides[key] = _parse_hparam_value(raw_value)
    if hparam_overrides:
        if not hasattr(model, "hparams") or not isinstance(model.hparams, dict):
            model.hparams = {}
        model.hparams.update(hparam_overrides)
        for key, value in hparam_overrides.items():
            if hasattr(model.config, key):
                setattr(model.config, key, value)
        sorted_pairs = ", ".join(
            f"{k}={hparam_overrides[k]}" for k in sorted(hparam_overrides.keys())
        )
        print(f"[hparam override] {sorted_pairs}")

    eval_pose_estimation(args, model, save_dir=args.output_dir)
