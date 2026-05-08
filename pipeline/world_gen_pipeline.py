"""
Prefect orchestration pipeline for full world generation.
Chains: scene setup -> render -> depth -> diffusion texture -> storage -> export.
SDKs: Prefect, MinIO, Apache Arrow, W&B
"""
import os
import time
import json
import uuid
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

import wandb


@dataclass
class WorldGenConfig:
    run_id: str = ""
    n_scenes: int = 10
    image_width: int = 1280
    image_height: int = 720
    n_objects_per_scene: int = 5
    generate_textures: bool = True
    estimate_depth: bool = True
    generate_meshes: bool = False
    texture_prompts: List[str] = None
    output_dir: str = "./world_gen_output"
    minio_endpoint: Optional[str] = None
    minio_bucket: str = "world-gen"
    wandb_project: str = "generative-world-engine"
    seed: int = 42

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"worldgen_{int(time.time())}"
        if self.texture_prompts is None:
            self.texture_prompts = [
                "rocky mountain terrain", "sandy desert ground",
                "green grass field", "wet concrete floor",
                "wooden planks", "metal grating",
            ]


@task(retries=2, retry_delay_seconds=5)
def generate_scenes(config: WorldGenConfig) -> List[Dict]:
    """Task: Generate synthetic scenes with Omniverse Replicator."""
    logger = get_run_logger()
    from replicator.scene_generator import OmniverseSceneGenerator

    gen = OmniverseSceneGenerator(
        output_dir=str(Path(config.output_dir) / "scenes"),
        image_width=config.image_width,
        image_height=config.image_height,
    )
    annotations = gen.generate_dataset(
        n_scenes=config.n_scenes,
        n_objects_range=(2, config.n_objects_per_scene + 3),
        seed=config.seed,
    )
    logger.info(f"Generated {len(annotations)} scenes")
    return [a.__dict__ if hasattr(a, "__dict__") else a for a in annotations]


@task
def generate_textures(config: WorldGenConfig) -> List[str]:
    """Task: Generate AI textures for scene surfaces."""
    logger = get_run_logger()
    try:
        from generation.diffusion_textures import TextureGenerator
        gen = TextureGenerator(device="cuda")
        output_dir = str(Path(config.output_dir) / "textures")
        images = gen.generate_texture_variants(
            base_prompt="photorealistic surface material",
            variants=config.texture_prompts,
            output_dir=output_dir,
            width=512, height=512, steps=25,
        )
        paths = [str(Path(output_dir) / f"texture_{i:03d}.png") for i in range(len(images))]
        logger.info(f"Generated {len(paths)} textures")
        return paths
    except Exception as e:
        logger.warning(f"Texture generation failed: {e}. Skipping.")
        return []


@task
def estimate_scene_depths(
    annotations: List[Dict], config: WorldGenConfig
) -> List[Dict]:
    """Task: Run depth estimation on rendered scenes."""
    logger = get_run_logger()
    try:
        from depth.depth_estimator import DepthEstimator
        estimator = DepthEstimator(device="cuda")
        depth_dir = str(Path(config.output_dir) / "depths")
        results = []
        for ann in annotations[:config.n_scenes]:
            img_path = ann.get("image_path", "")
            if img_path and Path(img_path).exists():
                stem = Path(img_path).stem
                out = str(Path(depth_dir) / f"{stem}_depth.png")
                estimator.estimate(img_path, output_path=out)
                results.append({"scene_id": ann.get("scene_id"), "depth_path": out})
        logger.info(f"Depth estimated for {len(results)} scenes")
        return results
    except Exception as e:
        logger.warning(f"Depth estimation failed: {e}. Skipping.")
        return []


@task
def export_arrow_dataset(
    annotations: List[Dict], config: WorldGenConfig
) -> str:
    """Task: Export scene metadata as Apache Arrow Parquet for ML pipelines."""
    rows = []
    for ann in annotations:
        for obj in ann.get("objects", []):
            rows.append({
                "scene_id": str(ann.get("scene_id", "")),
                "image_path": str(ann.get("image_path", "")),
                "obj_id": int(obj.get("id", 0)),
                "obj_class": str(obj.get("class", "")),
                "bbox_x1": float(obj.get("bbox_2d", [0,0,0,0])[0]),
                "bbox_y1": float(obj.get("bbox_2d", [0,0,0,0])[1]),
                "bbox_x2": float(obj.get("bbox_2d", [0,0,0,0])[2]),
                "bbox_y2": float(obj.get("bbox_2d", [0,0,0,0])[3]),
                "depth_z": float(obj.get("pose_3d", {}).get("z", 0)),
                "environment": str(ann.get("environment", "")),
            })

    if not rows:
        return ""

    schema = pa.schema([
        pa.field("scene_id", pa.string()),
        pa.field("image_path", pa.string()),
        pa.field("obj_id", pa.int32()),
        pa.field("obj_class", pa.string()),
        pa.field("bbox_x1", pa.float32()),
        pa.field("bbox_y1", pa.float32()),
        pa.field("bbox_x2", pa.float32()),
        pa.field("bbox_y2", pa.float32()),
        pa.field("depth_z", pa.float32()),
        pa.field("environment", pa.string()),
    ])

    table = pa.table(
        {k: [r[k] for r in rows] for k in rows[0].keys()},
        schema=schema,
    )

    out_path = str(Path(config.output_dir) / "dataset.parquet")
    pq.write_table(table, out_path, compression="snappy")
    get_run_logger().info(f"Arrow dataset: {len(rows)} records -> {out_path}")
    return out_path


@task
def upload_to_minio(file_paths: List[str], config: WorldGenConfig) -> List[str]:
    """Task: Upload generated assets to MinIO object storage."""
    if not MINIO_AVAILABLE or not config.minio_endpoint:
        return []
    logger = get_run_logger()
    client = Minio(config.minio_endpoint, secure=False)
    if not client.bucket_exists(config.minio_bucket):
        client.make_bucket(config.minio_bucket)

    uploaded = []
    for fpath in file_paths:
        if fpath and Path(fpath).exists():
            obj_name = f"{config.run_id}/{Path(fpath).name}"
            client.fput_object(config.minio_bucket, obj_name, fpath)
            url = f"http://{config.minio_endpoint}/{config.minio_bucket}/{obj_name}"
            uploaded.append(url)
            logger.info(f"Uploaded: {url}")
    return uploaded


@flow(name="world-gen-pipeline")
def world_gen_pipeline(config: WorldGenConfig = None) -> Dict[str, Any]:
    """
    Full world generation pipeline.
    Scenes -> Textures -> Depths -> Arrow Export -> MinIO Upload
    """
    config = config or WorldGenConfig()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    wb_run = wandb.init(
        project=config.wandb_project,
        name=config.run_id,
        config=config.__dict__,
        mode="online" if os.environ.get("WANDB_API_KEY") else "disabled",
    )

    annotations = generate_scenes(config)
    textures = generate_textures(config) if config.generate_textures else []
    depths = estimate_scene_depths(annotations, config) if config.estimate_depth else []
    parquet_path = export_arrow_dataset(annotations, config)

    all_files = (
        [a.get("image_path", "") for a in annotations] +
        textures +
        [parquet_path]
    )
    urls = upload_to_minio(all_files, config)

    summary = {
        "run_id": config.run_id,
        "n_scenes": len(annotations),
        "n_textures": len(textures),
        "n_depths": len(depths),
        "parquet_path": parquet_path,
        "minio_urls": urls,
    }

    wandb.log(summary)
    wb_run.finish()
    return summary
