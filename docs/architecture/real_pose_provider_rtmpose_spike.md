# Real Pose Provider RTMPose Spike

Date: 2026-07-23

## Official Compatibility Findings

Sources checked:

- MMPose installation: https://mmpose.readthedocs.io/en/latest/installation.html
- MMPose FAQ compatibility table: https://mmpose.readthedocs.io/en/latest/faq.html
- MMPose inference guide: https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md
- MMCV installation: https://github.com/open-mmlab/mmcv/blob/main/docs/en/get_started/installation.md
- PyTorch install selector: https://docs.pytorch.org/get-started/locally/
- OpenMMLab MMCV wheel index: https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

Compatibility matrix:

| Component | Official finding | Spike choice |
| --- | --- | --- |
| Python | MMPose says Python 3.7+; examples use conda Python 3.8. PyTorch latest requires Python 3.9+. | Python 3.11, because OpenMMLab publishes cp311 Windows wheels for MMCV/Torch 2.1. |
| PyTorch | MMPose requires PyTorch 1.8+. MMPose 1.0 advertised PyTorch 2.0 compatibility. | PyTorch 2.1.0 CPU first, because matching MMCV 2.1.0 Windows cp311 wheels exist. |
| MMCV | MMPose 1.3.2 requires mmcv>=2.0.1 and mmengine>=0.9.0. MMCV must match PyTorch/CUDA strictly. | mmcv==2.1.0 from OpenMMLab CPU torch2.1 wheel index. |
| MMEngine | MMPose 1.3.2 requires mmengine>=0.9.0. | mmengine==0.10.7. |
| MMPose | Latest PyPI release is 1.3.2. | mmpose==1.3.2. |
| MMDetection | MMPose inferencer top-down human detection may need mmdet. MMPose 1.x corresponds to mmdet 3.x and mmcv 2.x. | mmdet==3.2.0 to keep mmcv<2.2 compatibility. |
| Model | MMPose aliases include `human`, `body26`, and `wholebody`. | Start with `human` RTMPose-m for body/wrist speed; evaluate `body26` and `wholebody` only if wrist quality is insufficient. |

Current main environment is not a good target:

- Python 3.13.9.
- Torch metadata 2.12.0.
- Torch import fails with duplicate OpenMP runtime (`libiomp5md.dll already initialized`).
- mmengine/mmcv/mmpose/mmdet are not installed.

## Isolated Environment

Conda creation was attempted:

```powershell
conda create -n cricvision_pose_rtmpose python=3.11 -y
```

It failed on SSL certificate verification against `repo.anaconda.com`. I did not disable SSL verification.

A workspace-local venv was created successfully:

```powershell
conda run -n ai_env python -m venv .venv_pose
```

Official PyTorch wheel install was attempted:

```powershell
.\.venv_pose\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

It failed on SSL certificate verification against `download.pytorch.org`. I did not use `--trusted-host`.

Once local certificate trust is fixed, run:

```powershell
.\scripts\setup_rtmpose_pose_env.ps1
```

## Provider Implementation

Implemented:

- `Backends/src/release_point/rtmpose_provider.py`
- Lazy `RTMPoseProvider` using MMPose `MMPoseInferencer`.
- Normalization to existing `PoseProvider` contract.
- Provider metadata: `rtmpose_mmpose`, model alias/name, `coco17`.
- Clean-video enforcement remains in base `PoseProvider.estimate_sequence`.
- Bowling-arm heuristic from temporal wrist activity.
- Service integration behind `CRICVISION_RELEASE_POSE_PROVIDER=rtmpose`.

Fallback remains:

- Provider absent or failed -> `pose_provider_unavailable`, `pose_not_run`, trajectory-only fallback.
- Fake provider remains rejected in production.

## Real-Video Validation

Available clean CricVision raw clips exist under:

```text
outputs/video_analysis/<analysis_id>/raw/original_video.mp4
```

Run validation after RTMPose environment install:

```powershell
$env:CRICVISION_RELEASE_POSE_PROVIDER="rtmpose"
.\.venv_pose\Scripts\python.exe scripts\validate_rtmpose_provider.py analysis_20260718_065149_af258b --device cpu
```

Expected artifacts:

```text
outputs/video_analysis/<analysis_id>/reports/rtmpose_validation.json
outputs/video_analysis/<analysis_id>/reports/rtmpose_debug_frames/frame_*.jpg
```

Do not report REAL POSE PROVIDER READY until a real model loads, real clean frames infer, bowler selection is correct, wrist/arm keypoints are usable around release, and at least one Release analysis uses genuine pose evidence end-to-end.
