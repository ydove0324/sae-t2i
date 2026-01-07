
pip install uv
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://bytedpypi.byted.org/simple/
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb diffusers --index-url https://bytedpypi.byted.org/simple/
uv pip install "numpy<2" transformers einops omegaconf --index-url https://bytedpypi.byted.org/simple/
