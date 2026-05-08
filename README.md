# Generative World Engine

Cluster 10 of the NextAura 500 SDKs / 25 Clusters project.

Procedural world generation using physics, AI, and photorealistic rendering. Generate infinite synthetic datasets, 3D environments, and training scenes on demand.

## Architecture

- Omniverse Replicator for scene variation and synthetic data generation
- OpenUSD for scene description and asset interchange
- Gaussian Splatting + nerfstudio for real-world scene capture and reconstruction
- NVIDIA Warp for GPU physics in generated worlds
- HuggingFace Diffusers + ControlNet for AI-driven texture and asset generation
- Blender Python API for procedural mesh generation
- DALI for GPU-accelerated training data pipeline
- Depth Pro for monocular depth estimation
- MinIO for artifact storage, Prefect for pipeline orchestration
- W&B for experiment tracking

## SDKs Used

NVIDIA Cosmos SDK, NVIDIA Omniverse SDK, OpenUSD, Gaussian Splatting, nerfstudio, NVIDIA Warp, NVIDIA PhysX SDK, Omniverse Replicator, HuggingFace Diffusers, ControlNet, DALI, Blender Python API, OpenCV SDK, Depth Pro SDK, PyTorch, FastAPI, MinIO SDK, Prefect SDK, Weights & Biases, Apache Arrow

## Quickstart

```bash
pip install -r requirements.txt
python main.py --mode replicator --scenes 10 --output ./data
python main.py --mode diffuse --prompt "rocky mountain terrain at dusk" --output ./textures
python main.py --mode reconstruct --images ./captures --output ./splat
python main.py --mode pipeline --config configs/world_gen.yaml
```

## Structure

```
replicator/     Omniverse Replicator scene variation and annotation
usd/            OpenUSD scene construction and asset management
generation/     Diffusers + ControlNet texture and asset generation
reconstruction/ Gaussian Splatting + nerfstudio scene reconstruction
blender/        Blender Python API procedural mesh generation
depth/          Depth Pro monocular depth estimation
pipeline/       Prefect orchestration of full world-gen pipeline
storage/        MinIO artifact storage + Arrow data export
main.py         Entry point
```
