"""
generative-world-engine — Entry Point

Procedural world generation: synthetic scenes, AI textures, depth maps,
USD stages, and full Prefect pipeline orchestration.

Usage:
  python main.py --mode replicator --scenes 10 --output ./data
  python main.py --mode diffuse --prompt "rocky mountain terrain at dusk"
  python main.py --mode usd --objects 15 --output ./scene.usda
  python main.py --mode depth --input ./image.jpg
  python main.py --mode pipeline --scenes 20 --output ./world_data
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generative World Engine")
    parser.add_argument("--mode", required=True,
                        choices=["replicator", "diffuse", "usd", "depth", "pipeline", "blender"])
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--objects", type=int, default=5)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="rocky desert terrain at golden hour")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--minio", type=str, default=None, help="MinIO endpoint e.g. localhost:9000")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Generative World Engine")
    print(f"  Mode: {args.mode.upper()} | Device: {args.device}")
    print("=" * 60)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.mode == "replicator":
        from replicator.scene_generator import OmniverseSceneGenerator
        gen = OmniverseSceneGenerator(
            output_dir=args.output,
            image_width=args.width,
            image_height=args.height,
        )
        annotations = gen.generate_dataset(
            n_scenes=args.scenes,
            n_objects_range=(2, args.objects + 3),
            seed=args.seed,
        )
        print(f"
Generated {len(annotations)} scenes -> {args.output}")

    elif args.mode == "diffuse":
        from generation.diffusion_textures import TextureGenerator
        gen = TextureGenerator(device=args.device)
        out = str(Path(args.output) / "texture_000.png")
        img = gen.generate_texture(args.prompt, output_path=out, seed=args.seed)
        print(f"
Texture saved: {out} ({img.width}x{img.height})")

    elif args.mode == "usd":
        from usd.scene_builder import USDSceneBuilder
        out_path = args.output if args.output.endswith(".usda") else str(Path(args.output) / "scene.usda")
        builder = USDSceneBuilder(stage_path=out_path)
        builder.populate_random_scene(n_objects=args.objects, seed=args.seed)
        builder.save()
        print(f"
USD stage saved: {out_path}")

    elif args.mode == "depth":
        if not args.input:
            print("--input required for depth mode"); sys.exit(1)
        from depth.depth_estimator import DepthEstimator
        est = DepthEstimator(device=args.device)
        out = str(Path(args.output) / "depth.png")
        result = est.estimate(args.input, output_path=out, colorize=True)
        print(f"
Depth estimated -> {out}")
        print(f"  Shape: {result['depth'].shape}, metric: {result.get('metric', False)}")

    elif args.mode == "blender":
        from blender_gen.procedural_mesh import BlenderMeshGenerator
        gen = BlenderMeshGenerator()
        out = str(Path(args.output) / "scene.glb")
        gen.generate(scene_type="terrain", output_path=out, seed=args.seed)
        print(f"
Blender scene exported: {out}")

    elif args.mode == "pipeline":
        from pipeline.world_gen_pipeline import world_gen_pipeline, WorldGenConfig
        cfg = WorldGenConfig(
            n_scenes=args.scenes,
            image_width=args.width,
            image_height=args.height,
            n_objects_per_scene=args.objects,
            output_dir=args.output,
            minio_endpoint=args.minio,
            seed=args.seed,
        )
        result = world_gen_pipeline(cfg)
        print(f"
Pipeline complete:")
        print(f"  Scenes: {result['n_scenes']}")
        print(f"  Textures: {result['n_textures']}")
        print(f"  Parquet: {result['parquet_path']}")


if __name__ == "__main__":
    main()
