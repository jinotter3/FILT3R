import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import ast
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import gc
from datetime import timedelta
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data._utils.collate import default_collate
import tempfile
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


def _save_scene_arrays_npz(path_prefix: str, **arrays):
    """Save large scene tensors without Python pickle size limits."""
    np.savez(f"{path_prefix}.npz", **arrays)


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="max frames limit")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="7scenes",
        choices=["7scenes", "nrgbd", "long3d"],
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="dataset root directory (defaults depend on --eval_dataset)",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="optional single scene to evaluate",
    )
    parser.add_argument(
        "--kf_every",
        type=int,
        default=None,
        help="optional keyframe stride override",
    )
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
    parser.add_argument("--voxel_size", type=float, default=0.0, help="voxel size for voxel grid downsampling, 0 means no downsampling")
    parser.add_argument(
        "--eval_center_crop",
        type=int,
        default=0,
        help="optional square center-crop size for evaluation (0 disables crop and uses full frame)",
    )
    parser.add_argument(
        "--dist_timeout_min",
        type=int,
        default=120,
        help="distributed process-group timeout in minutes",
    )
    parser.add_argument(
        "--inference_impl",
        type=str,
        default="recurrent_lighter",
        choices=["full", "recurrent_lighter"],
        help="model forward implementation for evaluation; recurrent_lighter uses much less GPU memory",
    )
    parser.add_argument(
        "--icp_max_points",
        type=int,
        default=0,
        help="deprecated; kept for CLI compatibility and ignored",
    )
    parser.add_argument(
        "--icp_max_iter",
        type=int,
        default=0,
        help="deprecated; kept for CLI compatibility and ignored",
    )
    parser.add_argument(
        "--eval_sample_seed",
        type=int,
        default=0,
        help="deprecated; kept for CLI compatibility and ignored",
    )
    return parser


def main(args):
    if args.eval_center_crop < 0:
        raise ValueError("--eval_center_crop must be >= 0")
    if args.dist_timeout_min <= 0:
        raise ValueError("--dist_timeout_min must be > 0")

    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, Long3D
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    if args.eval_center_crop == 0:
        print("Evaluation crop: full frame")
    else:
        print(f"Evaluation crop: center {args.eval_center_crop}x{args.eval_center_crop}")
    print(f"Inference implementation: {args.inference_impl}")
    eval_dataset = args.eval_dataset.lower()
    default_roots = {
        "7scenes": "./data/7scenes",
        "nrgbd": "./data/NRGBD",
        "long3d": "./data/Long3D",
    }
    data_root = args.data_root or default_roots[eval_dataset]
    if not osp.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root does not exist: {data_root}. "
            f"Please pass a valid path via --data_root."
        )

    dataset_kwargs = dict(
        split="test",
        ROOT=data_root,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        max_frames=args.max_frames,
    )
    if args.scene_id is not None:
        dataset_kwargs["test_id"] = args.scene_id
    if args.kf_every is not None:
        dataset_kwargs["kf_every"] = args.kf_every

    if eval_dataset == "7scenes":
        dataset_kwargs.setdefault("kf_every", 2)
        dataset_name = "7scenes"
        dataset = SevenScenes(**dataset_kwargs)
    elif eval_dataset == "nrgbd":
        dataset_kwargs.setdefault("kf_every", 1)
        dataset_name = "NRGBD"
        dataset = NRGBD(**dataset_kwargs)
    elif eval_dataset == "long3d":
        dataset_kwargs.setdefault("kf_every", 1)
        dataset_name = "Long3D"
        dataset = Long3D(**dataset_kwargs)
    else:
        raise NotImplementedError(f"Unsupported dataset: {eval_dataset}")

    if len(dataset.scene_list) == 0:
        raise ValueError(
            f"No scenes found for dataset={eval_dataset} under root={data_root} "
            f"with scene_id={args.scene_id}."
        )

    requires_gt = eval_dataset != "long3d"
    print(f"Dataset: {dataset_name}")
    print(f"Dataset root: {data_root}")
    datasets_all = {dataset_name: dataset}

    # ====== print the number of views for each scene ======
    print("\n=== number of views for each scene ===")
    for name_data, dataset in datasets_all.items():
        print(f"\n{name_data} dataset:")
        for scene_id in dataset.scene_list:
            if eval_dataset == "nrgbd":
                # NRGBD dataset file structure
                if hasattr(dataset, "get_aligned_frame_ids"):
                    frame_ids = dataset.get_aligned_frame_ids(scene_id)
                    view_count = len(frame_ids[:: max(1, dataset.kf_every)])
                else:
                    data_path = osp.join(dataset.ROOT, scene_id, "images")
                    num_files = len(
                        [name for name in os.listdir(data_path) if name.endswith(".png")]
                    )
                    view_count = len(
                        [f"{i}" for i in range(num_files)][:: max(1, dataset.kf_every)]
                    )
            elif eval_dataset == "long3d":
                data_path = osp.join(dataset.ROOT, scene_id, "images", "scan_images")
                image_names = [
                    name
                    for name in os.listdir(data_path)
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                image_names = sorted(image_names, key=dataset._frame_sort_key)
                view_count = len(image_names[:: dataset.kf_every])
            else:
                # SevenScenes dataset file structure
                data_path = osp.join(dataset.ROOT, scene_id)
                num_files = len([name for name in os.listdir(data_path) if "color" in name])
                view_count = len([f"{i:06d}" for i in range(num_files)][::dataset.kf_every])
            
            # consider max_frames limit
            if dataset.max_frames is not None:
                actual_view_count = min(view_count, dataset.max_frames)
                print(f"  {scene_id}: {actual_view_count} views (original: {view_count}, limit: {dataset.max_frames})")
            else:
                print(f"  {scene_id}: {view_count} views")
    print("================================\n")
    # ====== print end ======

    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=args.dist_timeout_min))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    model_name = args.model_name
    # if model_name == "ours" or model_name == "cut3r":
    from dust3r.model import ARCroco3DStereo
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21, get_pred_pts3d
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
    if str(device) == "cpu":
        model.float()
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

    model.eval()
    # else:
    #     raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = (
        Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
        if requires_gt
        else None
    )

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    if args.inference_impl == "full":
                        ignore_keys = set(
                            [
                                "depthmap",
                                "dataset",
                                "label",
                                "instance",
                                "idx",
                                "true_shape",
                                "rng",
                            ]
                        )
                        for view in batch:
                            for name in view.keys():  # pseudo_focal
                                if name in ignore_keys:
                                    continue
                                if isinstance(view[name], tuple) or isinstance(
                                    view[name], list
                                ):
                                    view[name] = [
                                        x.to(device, non_blocking=True) for x in view[name]
                                    ]
                                else:
                                    view[name] = view[name].to(device, non_blocking=True)

                    # if model_name == "ours" or model_name == "cut3r":
                    revisit = args.revisit
                    update = not args.freeze
                    if revisit > 1:
                        # repeat input for 'revisit' times
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i)
                                    for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i)
                                    for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0:
                                    if not update:
                                        new_view["update"] = torch.zeros_like(
                                            batch[i]["update"]
                                        ).bool()
                                new_views.append(new_view)
                        batch = new_views
                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        start = time.time()
                        if args.inference_impl == "recurrent_lighter":
                            preds, batch = model.forward_recurrent_lighter(batch, device=device)
                        else:
                            output = model(batch)
                            preds, batch = output.ress, output.views
                            del output
                        end = time.time()
                    valid_length = len(preds) // revisit
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]
                    fps = len(batch) / (end - start)
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                    )
                    # continue
                    fps_all.append(fps)
                    time_all.append(end - start)

                    # Evaluation / export
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    if requires_gt:
                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )
                        pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                            monitoring["pred_scale"],
                            monitoring["gt_scale"],
                            monitoring["pred_shift_z"],
                            monitoring["gt_shift_z"],
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            pts = pred_pts[j].cpu().numpy()[0]
                            conf = preds[j]["conf"].cpu().data.numpy()[0]

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                            if args.eval_center_crop > 0:
                                H, W = image.shape[:2]
                                crop_size = args.eval_center_crop
                                if crop_size > min(H, W):
                                    raise ValueError(
                                        f"eval_center_crop={crop_size} exceeds image size {(W, H)}"
                                    )
                                l = (W - crop_size) // 2
                                t = (H - crop_size) // 2
                                r = l + crop_size
                                b = t + crop_size
                                image = image[t:b, l:r]
                                mask = mask[t:b, l:r]
                                pts = pts[t:b, l:r]
                                pts_gt = pts_gt[t:b, l:r]
                                conf = conf[t:b, l:r]

                            # Align predicted 3D points to the ground truth.
                            pts[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts = geotrf(in_camera1, pts)

                            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append((image[None, ...] + 1.0) / 2.0)
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])

                        images_all = np.concatenate(images_all, axis=0)
                        pts_all = np.concatenate(pts_all, axis=0)
                        pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                        masks_all = np.concatenate(masks_all, axis=0)

                        scene_id = view["label"][0].rsplit("/", 1)[0]

                        _save_scene_arrays_npz(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}"),
                            images_all=images_all,
                            pts_all=pts_all,
                            pts_gt_all=pts_gt_all,
                            masks_all=masks_all,
                        )

                        if "DTU" in name_data:
                            threshold = 100
                        else:
                            threshold = 0.1

                        pts_all_masked = pts_all[masks_all > 0]
                        pts_gt_all_masked = pts_gt_all[masks_all > 0]
                        images_all_masked = images_all[masks_all > 0]

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            pts_all_masked.reshape(-1, 3)
                        )
                        pcd.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(
                            pts_gt_all_masked.reshape(-1, 3)
                        )
                        pcd_gt.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )

                        # ====== voxel grid downsampling ======
                        if args.voxel_size > 0:
                            pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                            pcd_gt = pcd_gt.voxel_down_sample(voxel_size=args.voxel_size)
                        # ===========================

                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                            ),
                            pcd,
                        )

                        o3d.io.write_point_cloud(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                            pcd_gt,
                        )

                        trans_init = np.eye(4)

                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd,
                            pcd_gt,
                            threshold,
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )

                        transformation = reg_p2p.transformation

                        pcd = pcd.transform(transformation)
                        pcd.estimate_normals()
                        pcd_gt.estimate_normals()

                        gt_normal = np.asarray(pcd_gt.normals)
                        pred_normal = np.asarray(pcd.normals)

                        acc, acc_med, nc1, nc1_med = accuracy(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        comp, comp_med, nc2, nc2_med = completion(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        print(
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                        )
                        print(
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                            file=open(log_file, "a"),
                        )

                        acc_all += acc
                        comp_all += comp
                        nc1_all += nc1
                        nc2_all += nc2

                        acc_all_med += acc_med
                        comp_all_med += comp_med
                        nc1_all_med += nc1_med
                        nc2_all_med += nc2_med

                        # release cuda memory
                        del (
                            gt_pts,
                            pred_pts,
                            gt_factor,
                            pr_factor,
                            masks,
                            monitoring,
                            images_all,
                            pts_all,
                            pts_gt_all,
                            masks_all,
                            pts_all_masked,
                            pts_gt_all_masked,
                            images_all_masked,
                            pcd,
                            pcd_gt,
                            preds,
                            batch,
                        )
                    else:
                        pts_all = []
                        images_all = []
                        masks_all = []

                        for j, view in enumerate(batch):
                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            pts = get_pred_pts3d(view, preds[j], use_pose=True).cpu().numpy()[0]
                            conf = preds[j]["conf"].cpu().data.numpy()[0]
                            mask = conf > args.conf_thresh

                            if args.eval_center_crop > 0:
                                H, W = image.shape[:2]
                                crop_size = args.eval_center_crop
                                if crop_size > min(H, W):
                                    raise ValueError(
                                        f"eval_center_crop={crop_size} exceeds image size {(W, H)}"
                                    )
                                l = (W - crop_size) // 2
                                t = (H - crop_size) // 2
                                r = l + crop_size
                                b = t + crop_size
                                image = image[t:b, l:r]
                                pts = pts[t:b, l:r]
                                mask = mask[t:b, l:r]

                            images_all.append((image[None, ...] + 1.0) / 2.0)
                            pts_all.append(pts[None, ...])
                            masks_all.append(mask[None, ...])

                        images_all = np.concatenate(images_all, axis=0)
                        pts_all = np.concatenate(pts_all, axis=0)
                        masks_all = np.concatenate(masks_all, axis=0)
                        scene_id = batch[-1]["label"][0].rsplit("/", 1)[0]

                        _save_scene_arrays_npz(
                            os.path.join(save_path, f"{scene_id.replace('/', '_')}"),
                            images_all=images_all,
                            pts_all=pts_all,
                            masks_all=masks_all,
                        )

                        pts_all_masked = pts_all[masks_all > 0]
                        images_all_masked = images_all[masks_all > 0]
                        if pts_all_masked.shape[0] == 0:
                            warn_str = (
                                f"Idx: {scene_id}, empty point cloud after conf "
                                f"threshold {args.conf_thresh}"
                            )
                            print(warn_str)
                            print(warn_str, file=open(log_file, "a"))
                            del images_all, pts_all, masks_all, preds, batch
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            pts_all_masked.reshape(-1, 3)
                        )
                        pcd.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        if args.voxel_size > 0:
                            pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

                        pred_ply_path = os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-pred.ply"
                        )
                        o3d.io.write_point_cloud(pred_ply_path, pcd)

                        gt_cloud_path = osp.join(dataset.ROOT, scene_id, "dense_cloud_map.pcd")
                        if osp.isfile(gt_cloud_path):
                            gt_pcd = o3d.io.read_point_cloud(gt_cloud_path)
                            if args.voxel_size > 0:
                                gt_pcd = gt_pcd.voxel_down_sample(
                                    voxel_size=args.voxel_size
                                )
                            o3d.io.write_point_cloud(
                                os.path.join(
                                    save_path, f"{scene_id.replace('/', '_')}-gt.ply"
                                ),
                                gt_pcd,
                            )
                            del gt_pcd

                        export_str = (
                            f"Idx: {scene_id}, Saved: {pred_ply_path}, "
                            f"Points: {len(pcd.points)}"
                        )
                        print(export_str)
                        print(export_str, file=open(log_file, "a"))

                        del (
                            images_all,
                            pts_all,
                            masks_all,
                            pts_all_masked,
                            images_all_masked,
                            pcd,
                            preds,
                            batch,
                        )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    if len(metrics) > 0:
                        mean_metrics = {
                            metric: sum(values) / len(values)
                            for metric, values in metrics.items()
                        }

                        c_name = "mean"
                        print_str = f"{c_name.ljust(20)}: "
                        for m_name in mean_metrics:
                            print_num = np.mean(mean_metrics[m_name])
                            print_str = print_str + f"{m_name}: {print_num:.3f} | "
                        print_str = print_str + "\n"
                        f.write(to_write + print_str)
                    else:
                        f.write(to_write)
                        if to_write and not to_write.endswith("\n"):
                            f.write("\n")
                        if not to_write:
                            f.write("No metric entries found.\n")


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
