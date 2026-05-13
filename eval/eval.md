# Evaluation

We provide long-evaluation entry scripts per dataset:

- `eval/relpose/run_tum.sh`
- `eval/relpose/run_scannet.sh`
- `eval/video_depth/run_bonn.sh`
- `eval/video_depth/run_kitti.sh`
- `eval/mv_recon/run.sh`
- `eval/mv_recon/run_nrgbd.sh`

All results are written under `eval_results/` by default.

## Dataset roots

By default the evaluation code looks for datasets under `data/`. You can either place them there or override the roots with environment variables.

Supported environment variables:

- `KITTI_ROOT`
- `BONN_ROOT`
- `SCANNET_ROOT`
- `TUM_ROOT`
- `SINTEL_ROOT`
- `SINTEL_CAM_ROOT`
- `CONTEXT_MEMORY_FRAMES`
- `CONTEXT_MEMORY_JSONS`

Default locations used by the public long-horizon scripts:

```text
data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_<N>
data/long_bonn_s1/rgbd_bonn_dataset
data/long_scannet_s3
data/long_tum_s1
```

For multi-view reconstruction, `eval/mv_recon/launch.py` defaults to:

```text
data/7scenes
data/NRGBD
data/Long3D
```

and you can override them with `DATA_ROOT=...` in the wrapper script.

For acquiring the raw datasets, follow the dataset instructions from [TTT3R](https://github.com/Inception3D/TTT3R). FILT3R includes small preprocessing helpers for the long-horizon layouts expected by the wrappers:

```bash
python datasets_preprocess/long_prepare_tum.py \
  --input-root /path/to/tum \
  --output-root data/long_tum_s1

python datasets_preprocess/long_prepare_bonn.py \
  --input-root /path/to/rgbd_bonn_dataset \
  --output-root data/long_bonn_s1/rgbd_bonn_dataset

python datasets_preprocess/long_prepare_scannet.py \
  --input-root /path/to/scannetv2 \
  --output-root data/long_scannet_s3 \
  --sample-interval 3

python datasets_preprocess/long_prepare_kitti.py \
  --input-root /path/to/kitti/val \
  --output-root data/long_kitti_s1/depth_selection/val_selection_cropped
```

## Camera Pose

Public long-evaluation wrappers:

- `eval/relpose/run_tum.sh` defaults to `tum_s1_800`
- `eval/relpose/run_scannet.sh` defaults to `scannet_s3_50 scannet_s3_100 scannet_s3_150 scannet_s3_200 scannet_s3_300 scannet_s3_400 scannet_s3_500 scannet_s3_600 scannet_s3_700 scannet_s3_800 scannet_s3_900 scannet_s3_1000`
- models: `cut3r ttt3r filt3r`

Example TUM run:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_NAMES="cut3r ttt3r filt3r" \
MODEL_WEIGHTS=src/cut3r_512_dpt_4_64.pth \
bash eval/relpose/run_tum.sh
```

Example ScanNet run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SCANNET_DATASETS="scannet_s3_300 scannet_s3_600 scannet_s3_1000" \
bash eval/relpose/run_scannet.sh
```

Useful overrides:

- `TUM_DATASETS="tum_s1_600 tum_s1_800"`
- `SCANNET_DATASETS="scannet_s3_300 scannet_s3_600"`
- `SEQ_LIST="rgbd_dataset_freiburg3_sitting_halfsphere"`
- `FULL_SEQ=true`
- `NUM_PROCESSES=2`
- `MAIN_PROCESS_PORT=29551`
- `RUN_TAG=my_run`
- `OVERWRITE=true`
- `EXTRA_MODEL_HPARAMS="kalman_fixed_r=0.8"`

## Video Depth

Public long-evaluation wrappers:

- `eval/video_depth/run_bonn.sh` defaults to `bonn_s1_500`
- `eval/video_depth/run_kitti.sh` defaults to `kitti_s1_50 kitti_s1_100 kitti_s1_200 kitti_s1_400`
- models: `cut3r ttt3r filt3r`

Example Bonn run:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_NAMES="cut3r ttt3r filt3r" \
MODEL_WEIGHTS=src/cut3r_512_dpt_4_64.pth \
bash eval/video_depth/run_bonn.sh
```

Example KITTI run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
KITTI_DATASETS="kitti_s1_100 kitti_s1_400" \
bash eval/video_depth/run_kitti.sh
```

Both wrappers run inference first, then evaluate:

- `metric`
- `scale`
- `scale&shift`

Useful overrides:

- `BONN_DATASETS="bonn_s1_300 bonn_s1_500"`
- `KITTI_DATASETS="kitti_s1_100 kitti_s1_400"`
- `SEQ_LIST="balloon2 crowd2"`
- `DEPTH_ALIGNS="metric scale scale&shift"`
- `FULL_SEQ=true`
- `NUM_PROCESSES=2`
- `MAIN_PROCESS_PORT=29556`
- `RUN_TAG=my_run`
- `OVERWRITE=true`
- `EXTRA_MODEL_HPARAMS="kalman_fixed_r=0.8"`

## 3D Reconstruction

Public long-evaluation wrappers:

- `eval/mv_recon/run.sh` evaluates `7scenes` with frame budgets `300 400 500` by default
- `eval/mv_recon/run_nrgbd.sh` evaluates `nrgbd` with frame budget `1000` by default
- models: `cut3r ttt3r filt3r`

Example 7Scenes run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_NAMES="cut3r ttt3r filt3r" \
MODEL_WEIGHTS=src/cut3r_512_dpt_4_64.pth \
FRAME_BUDGETS="300 500" \
bash eval/mv_recon/run.sh
```

Example NRGBD run:

```bash
CUDA_VISIBLE_DEVICES=0 \
NRGBD_ROOT=/path/to/NRGBD \
NRGBD_MAX_FRAMES_LIST="500 1000" \
bash eval/mv_recon/run_nrgbd.sh
```

Useful overrides:

- `DATA_ROOT=/path/to/7scenes`
- `SCENE_ID=chess`
- `FRAME_BUDGETS="300 400 500"`
- `MAX_FRAMES=500`
- `NUM_PROCESSES=2`
- `MAIN_PROCESS_PORT=29502`
- `RUN_TAG=my_run`
- `OVERWRITE=true`
- `EXTRA_MODEL_HPARAMS="kalman_fixed_r=0.8"`
- `NRGBD_ROOT=/path/to/NRGBD`
- `NRGBD_SCENE=kitchen`
- `NRGBD_KF_EVERY=1`
- `NRGBD_MAX_FRAMES_LIST="500 1000"`

For other multi-view datasets such as Long3D, call `eval/mv_recon/launch.py` directly.
