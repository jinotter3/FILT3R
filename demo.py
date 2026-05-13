#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import argparse
import ast
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import datetime
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from einops import rearrange
from typing import List, Optional

try:
    import cv2
except Exception:
    cv2 = None

# Set random seed for reproducibility.
random.seed(42)

framerate = 30

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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the point cloud viewer",
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
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval for video processing (e.g., 1 means every frame, 2 means every other frame)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum number of frames/images to process after frame_interval subsampling; <=0 means no limit",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for the point cloud viewer",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Skip launching the interactive viewer (useful for batch diagnostics generation).",
    )
    parser.add_argument(
        "--model_hparam",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Generic model hparam override (repeatable), e.g. --model_hparam kalman_p_init=1.5. "
            "For filt3r, built-in defaults are applied first and these overrides win."
        ),
    )
    parser.add_argument(
        "--sky_seg",
        action="store_true",
        help="Enable learned sky segmentation and suppress sky points in viewer/exported confidence maps.",
    )
    parser.add_argument(
        "--sky_seg_model",
        type=str,
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        help="HuggingFace model id for semantic sky segmentation.",
    )
    parser.add_argument(
        "--sky_seg_device",
        type=str,
        default="cpu",
        help="Device for sky segmentation model (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--sky_seg_batch_size",
        type=int,
        default=8,
        help="Batch size for sky segmentation inference.",
    )
    parser.add_argument(
        "--sky_conf_value",
        type=float,
        default=0.0,
        help="Confidence value assigned to sky pixels before viewer thresholding.",
    )
    parser.add_argument(
        "--sky_seg_save_mask",
        action="store_true",
        help="Save predicted sky masks to output_dir/sky_mask as debug artifacts.",
    )
    return parser.parse_args()


def _resolve_torch_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[sky-seg] Requested device '{device}' not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _find_sky_class_id(id2label) -> int:
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            if str(v).strip().lower() == "sky":
                try:
                    return int(k)
                except Exception:
                    pass
    # Cityscapes default sky id.
    return 10


class SkySegmenter:
    def __init__(self, model_id: str, device: str = "cpu"):
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
            AutoModelForUniversalSegmentation,
        )

        self.model_id = model_id
        self.device = _resolve_torch_device(device)
        print(f"[sky-seg] Loading {model_id} on {self.device} ...")
        cfg = AutoConfig.from_pretrained(model_id)
        self.model_type = str(getattr(cfg, "model_type", "")).lower()
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        universal_model_types = {"detr", "eomt", "mask2former", "maskformer", "oneformer"}
        model_kind = (
            "universal" if self.model_type in universal_model_types else "semantic"
        )
        model_cls = (
            AutoModelForUniversalSegmentation
            if model_kind == "universal"
            else AutoModelForSemanticSegmentation
        )

        # Force safetensors loading so this still works on torch<2.6 where
        # transformers blocks .bin loading due CVE-2025-32434.
        load_kwargs = {"use_safetensors": True}
        try:
            self.model = model_cls.from_pretrained(model_id, **load_kwargs)
        except Exception as e:
            msg = str(e)
            if (
                "model.safetensors" in msg
                or "safetensors" in msg
                or "torch.load" in msg
                or "CVE-2025-32434" in msg
            ):
                raise RuntimeError(
                    "[sky-seg] Failed to load model in safetensors-only mode. "
                    "This environment blocks .bin checkpoint loading with torch<2.6. "
                    "Use a model repo that provides model.safetensors, or upgrade torch>=2.6."
                ) from e
            raise

        self.model.to(self.device).eval()
        self.sky_class_id = _find_sky_class_id(getattr(self.model.config, "id2label", {}))
        processor_name = self.processor.__class__.__name__.lower()
        self.requires_task_input = "oneformer" in processor_name
        print(
            f"[sky-seg] Loaded {model_kind} model (type={self.model_type}), "
            f"class id {self.sky_class_id} for sky."
        )

    @torch.no_grad()
    def predict_masks(
        self,
        images_rgb_u8: List[np.ndarray],
        batch_size: int = 8,
    ) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        if not images_rgb_u8:
            return masks
        batch_size = max(1, int(batch_size))
        for start in range(0, len(images_rgb_u8), batch_size):
            batch = images_rgb_u8[start : start + batch_size]
            target_sizes = [img.shape[:2] for img in batch]
            processor_kwargs = {"images": batch, "return_tensors": "pt"}
            if self.requires_task_input:
                processor_kwargs["task_inputs"] = ["semantic"] * len(batch)
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            seg_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            for seg in seg_maps:
                masks.append((seg.cpu().numpy() == self.sky_class_id))
        return masks


def _view_to_rgb_u8(view) -> np.ndarray:
    # view["img"]: [1, 3, H, W], normalized in [-1, 1]
    rgb = (0.5 * (view["img"][0].permute(1, 2, 0).cpu().numpy() + 1.0)).clip(0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def _suppress_confidence_with_masks(
    conf_list: List[torch.Tensor],
    masks: List[np.ndarray],
    conf_value: float,
) -> None:
    if len(conf_list) != len(masks):
        raise ValueError(
            f"Mask/conf length mismatch: masks={len(masks)} conf={len(conf_list)}"
        )
    for idx, (conf, mask_np) in enumerate(zip(conf_list, masks)):
        if conf.ndim == 3 and conf.shape[0] == 1:
            h, w = conf.shape[-2], conf.shape[-1]
            if mask_np.shape != (h, w):
                raise ValueError(
                    f"Mask shape mismatch at {idx}: got {mask_np.shape}, expected {(h, w)}"
                )
            mask = torch.from_numpy(mask_np).to(device=conf.device, dtype=torch.bool)
            conf[0][mask] = float(conf_value)
        elif conf.ndim == 2:
            h, w = conf.shape[-2], conf.shape[-1]
            if mask_np.shape != (h, w):
                raise ValueError(
                    f"Mask shape mismatch at {idx}: got {mask_np.shape}, expected {(h, w)}"
                )
            mask = torch.from_numpy(mask_np).to(device=conf.device, dtype=torch.bool)
            conf[mask] = float(conf_value)
        else:
            raise ValueError(
                f"Unsupported confidence tensor shape at {idx}: {tuple(conf.shape)}"
            )


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
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
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Combine images and raymaps.
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
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
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
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(
    outputs,
    outdir,
    revisit=1,
    use_pose=True,
    sky_segmenter: Optional[SkySegmenter] = None,
    sky_seg_batch_size: int = 8,
    sky_conf_value: float = 0.0,
    sky_seg_save_mask: bool = False,
):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod


    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    sky_masks = None
    if sky_segmenter is not None:
        print("[sky-seg] Predicting sky masks...")
        rgb_u8_views = [_view_to_rgb_u8(view) for view in outputs["views"]]
        sky_masks = sky_segmenter.predict_masks(
            rgb_u8_views, batch_size=sky_seg_batch_size
        )
        if len(sky_masks) != len(outputs["views"]):
            raise RuntimeError(
                f"[sky-seg] mask/view count mismatch: {len(sky_masks)} vs {len(outputs['views'])}"
            )
        sky_ratios = [float(mask.mean()) for mask in sky_masks]
        print(
            "[sky-seg] done. sky ratio "
            f"(mean/min/max): {np.mean(sky_ratios):.3f}/{np.min(sky_ratios):.3f}/{np.max(sky_ratios):.3f}"
        )

        if sky_seg_save_mask:
            sky_dir = os.path.join(outdir, "sky_mask")
            if os.path.exists(sky_dir):
                shutil.rmtree(sky_dir)
            os.makedirs(sky_dir, exist_ok=True)
            for i, mask in enumerate(sky_masks):
                iio.imwrite(os.path.join(sky_dir, f"{i:06d}.png"), mask.astype(np.uint8) * 255)

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    if sky_masks is not None:
        _suppress_confidence_with_masks(conf_self, sky_masks, sky_conf_value)
        _suppress_confidence_with_masks(conf_other, sky_masks, sky_conf_value)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    if os.path.exists(os.path.join(outdir, "depth")):
        shutil.rmtree(os.path.join(outdir, "depth"))
    if os.path.exists(os.path.join(outdir, "conf")):
        shutil.rmtree(os.path.join(outdir, "conf"))
    if os.path.exists(os.path.join(outdir, "color")):
        shutil.rmtree(os.path.join(outdir, "color"))
    if os.path.exists(os.path.join(outdir, "camera")):
        shutil.rmtree(os.path.join(outdir, "camera"))
    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)
    return pts3ds_other, colors, conf_other, cam_dict

def parse_seq_path(p, frame_interval=1, max_frames=500):
    global framerate
    
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = [path for path in all_img_paths 
                    if os.path.splitext(path.lower())[1] in img_extensions]
        
        excluded_suffixes = (".depth.png", ".depth.proj.png")
        img_paths = [
            path for path in img_paths if not path.lower().endswith(excluded_suffixes)
        ]


        if not img_paths:
            raise ValueError(f"No image files found in directory {p}")
        
        if frame_interval > 1:
            img_paths = img_paths[::frame_interval]
        if max_frames > 0:
            img_paths = img_paths[:max_frames]

        print(
            f" - Image sequence: Total images: {len(all_img_paths)}, "
            f"Frame interval: {frame_interval}, Max frames: {max_frames if max_frames > 0 else 'unlimited'}, "
            f"Images to process: {len(img_paths)}"
        )
        
        framerate = 30.0 / frame_interval
        
        tmpdirname = None
    else:
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required to read video inputs. "
                "Install with `pip install opencv-python`."
            )
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        
        framerate = video_fps / frame_interval
        
        frame_indices = list(range(0, total_frames, frame_interval))
        if max_frames > 0:
            frame_indices = frame_indices[:max_frames]
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, "
            f"Max frames: {max_frames if max_frames > 0 else 'unlimited'}, "
            f"Total Frames to Read: {len(frame_indices)}, Processed Framerate: {framerate}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval, args.max_frames)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    if str(device) == "cpu":
        model.float()
    model.config.model_update_type = args.model_update_type

    hparam_overrides = {}
    resolved_update_type = str(args.model_update_type)
    if resolved_update_type == "filt3r":
        hparam_overrides.update(FILT3R_DEFAULT_HPARAMS)

    for item in getattr(args, "model_hparam", []):
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

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference_recurrent_lighter(views, model, device)

    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    FPS_num = 1 / per_frame_time
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
    )

    sky_segmenter = None
    if getattr(args, "sky_seg", False):
        sky_segmenter = SkySegmenter(
            model_id=args.sky_seg_model,
            device=args.sky_seg_device,
        )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs,
        args.output_dir,
        1,
        True,
        sky_segmenter=sky_segmenter,
        sky_seg_batch_size=args.sky_seg_batch_size,
        sky_conf_value=args.sky_conf_value,
        sky_seg_save_mask=args.sky_seg_save_mask,
    )

    if getattr(args, "no_viewer", False):
        print("Skipping viewer (--no_viewer enabled).")
        return

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size,
        port = args.port,
        downsample_factor=args.downsample_factor
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
