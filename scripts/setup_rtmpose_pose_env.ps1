param(
    [string]$EnvPath = ".venv_pose",
    [string]$SeedCondaEnv = "ai_env"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvPath)) {
    conda run -n $SeedCondaEnv python -m venv $EnvPath
}

$python = Join-Path $EnvPath "Scripts/python.exe"

& $python -m pip install --upgrade pip --use-feature=truststore
& $python -m pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu --use-feature=truststore
& $python -m pip install numpy==1.26.4 setuptools==80.9.0 wheel pydantic==2.13.4 --use-feature=truststore
& $python -m pip install chumpy==0.70 --no-build-isolation --use-feature=truststore
& $python -m pip install mmengine==0.10.7 mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.2 opencv-python==4.10.0.84 numpy==1.26.4 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html --only-binary=mmcv --use-feature=truststore

& $python -c "import torch, mmengine, mmcv, mmpose, mmdet; print({'torch': torch.__version__, 'mmengine': mmengine.__version__, 'mmcv': mmcv.__version__, 'mmpose': mmpose.__version__, 'mmdet': mmdet.__version__})"
