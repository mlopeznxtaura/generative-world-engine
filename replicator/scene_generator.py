"""
Omniverse Replicator synthetic scene generation.
Generates N scene variations with PNG renders + JSON annotations.
SDKs: Omniverse Replicator, OpenUSD
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SceneAnnotation:
    scene_id: str
    image_path: str
    depth_path: str
    segmentation_path: str
    objects: List[Dict[str, Any]]   # [{id, class, bbox_2d, pose_3d, size}]
    camera: Dict[str, Any]          # {position, rotation, focal_length, fov}
    lighting: Dict[str, Any]        # {type, intensity, color, direction}
    environment: str                 # sky dome / hdri name


class OmniverseSceneGenerator:
    """
    Generate photorealistic synthetic scenes using Omniverse Replicator.
    Falls back to a headless procedural generator when Omniverse is unavailable.
    """

    OBJECT_CLASSES = [
        "box", "cylinder", "sphere", "cone", "torus",
        "car", "chair", "table", "person", "barrel",
    ]

    HDRI_ENVIRONMENTS = [
        "outdoor_sunny", "outdoor_cloudy", "indoor_warehouse",
        "indoor_office", "night_city", "sunset_desert",
    ]

    def __init__(
        self,
        output_dir: str = "./synthetic_data",
        use_omniverse: bool = False,
        image_width: int = 1280,
        image_height: int = 720,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_omniverse = use_omniverse
        self.W = image_width
        self.H = image_height

        if use_omniverse:
            self._init_omniverse()
        else:
            print("[Replicator] Running in procedural stub mode (Omniverse not available)")

    def _init_omniverse(self):
        """Initialize Omniverse kit and Replicator."""
        try:
            import omni.replicator.core as rep
            import omni.kit.app
            self.rep = rep
            print("[Replicator] Omniverse Replicator initialized")
        except ImportError:
            print("[Replicator] Omniverse not available — using procedural fallback")
            self.use_omniverse = False

    def generate_scene(self, scene_id: str, n_objects: int = 5, seed: int = 0) -> SceneAnnotation:
        """Generate a single scene with random objects, lighting, and camera."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        if self.use_omniverse:
            return self._generate_omniverse_scene(scene_id, n_objects, rng, np_rng)
        return self._generate_procedural_scene(scene_id, n_objects, rng, np_rng)

    def _generate_procedural_scene(
        self, scene_id: str, n_objects: int,
        rng: random.Random, np_rng: np.random.Generator
    ) -> SceneAnnotation:
        """Procedural fallback: generate metadata + render placeholder image."""
        import cv2

        # Render a simple synthetic image
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        depth = np.zeros((self.H, self.W), dtype=np.float32)
        seg = np.zeros((self.H, self.W), dtype=np.uint8)

        env = rng.choice(self.HDRI_ENVIRONMENTS)
        bg_color = {
            "outdoor_sunny": (135, 206, 235),
            "outdoor_cloudy": (180, 180, 200),
            "indoor_warehouse": (60, 60, 80),
            "indoor_office": (220, 200, 180),
            "night_city": (10, 10, 40),
            "sunset_desert": (255, 140, 60),
        }.get(env, (100, 100, 100))
        img[:] = bg_color

        objects = []
        for obj_id in range(n_objects):
            cls = rng.choice(self.OBJECT_CLASSES)
            cx = int(np_rng.uniform(0.1, 0.9) * self.W)
            cy = int(np_rng.uniform(0.2, 0.8) * self.H)
            w = int(np_rng.uniform(30, 120))
            h = int(np_rng.uniform(30, 120))
            color = tuple(int(c) for c in np_rng.integers(50, 255, 3))

            x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
            x2, y2 = min(self.W, cx + w // 2), min(self.H, cy + h // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Label
            cv2.putText(img, cls, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Depth: closer objects get smaller depth value
            z = float(np_rng.uniform(1.0, 10.0))
            depth[y1:y2, x1:x2] = z
            seg[y1:y2, x1:x2] = obj_id + 1

            objects.append({
                "id": obj_id,
                "class": cls,
                "bbox_2d": [x1, y1, x2, y2],
                "pose_3d": {"x": float(np_rng.uniform(-5, 5)),
                            "y": float(np_rng.uniform(-5, 5)),
                            "z": z},
                "size": {"w": x2 - x1, "h": y2 - y1},
            })

        # Save outputs
        img_path = str(self.output_dir / f"{scene_id}_rgb.png")
        depth_path = str(self.output_dir / f"{scene_id}_depth.png")
        seg_path = str(self.output_dir / f"{scene_id}_seg.png")

        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        depth_vis = (depth / depth.max() * 255).astype(np.uint8) if depth.max() > 0 else depth.astype(np.uint8)
        cv2.imwrite(depth_path, depth_vis)
        cv2.imwrite(seg_path, seg * 25)  # Scale for visibility

        cam_pos = np_rng.uniform(-2, 2, 3).tolist()
        return SceneAnnotation(
            scene_id=scene_id,
            image_path=img_path,
            depth_path=depth_path,
            segmentation_path=seg_path,
            objects=objects,
            camera={
                "position": cam_pos,
                "rotation": np_rng.uniform(-30, 30, 3).tolist(),
                "focal_length": float(np_rng.uniform(24, 85)),
                "fov": float(np_rng.uniform(45, 90)),
            },
            lighting={
                "type": rng.choice(["directional", "point", "area"]),
                "intensity": float(np_rng.uniform(500, 2000)),
                "color": np_rng.uniform(0.8, 1.0, 3).tolist(),
                "direction": np_rng.uniform(-1, 1, 3).tolist(),
            },
            environment=env,
        )

    def _generate_omniverse_scene(self, scene_id, n_objects, rng, np_rng):
        """Full Omniverse Replicator scene generation."""
        rep = self.rep
        with rep.new_layer():
            # Environment
            env = rng.choice(self.HDRI_ENVIRONMENTS)
            rep.create.sky(texture=f"omniverse://localhost/NVIDIA/Assets/Skies/{env}.hdr")

            # Camera
            camera = rep.create.camera(
                position=np_rng.uniform(-5, 5, 3).tolist(),
                look_at=(0, 0, 0),
            )

            # Objects
            shapes = []
            for i in range(n_objects):
                cls = rng.choice(["cube", "sphere", "cylinder", "cone"])
                shape = rep.create.__dict__[cls](
                    position=np_rng.uniform(-3, 3, 3).tolist(),
                    scale=float(np_rng.uniform(0.5, 2.0)),
                )
                shapes.append(shape)

            # Randomize materials
            with rep.randomizer.materials(shapes):
                rep.distribution.choice(["OmniPBR"])

            # Render
            render_product = rep.create.render_product(camera, (self.W, self.H))
            output = rep.WriterRegistry.get("BasicWriter")
            output.initialize(
                output_dir=str(self.output_dir),
                rgb=True, depth=True, semantic_segmentation=True,
            )
            output.attach([render_product])
            rep.orchestrator.run()

        return self.generate_scene(scene_id, n_objects, seed=0)  # Return stub annotation

    def generate_dataset(
        self,
        n_scenes: int = 10,
        n_objects_range: Tuple[int, int] = (3, 8),
        seed: int = 42,
    ) -> List[SceneAnnotation]:
        """Generate N scenes and save all annotations to JSON."""
        rng = random.Random(seed)
        annotations = []
        print(f"[Replicator] Generating {n_scenes} scenes...")
        for i in range(n_scenes):
            n_obj = rng.randint(*n_objects_range)
            scene_id = f"scene_{i:04d}"
            ann = self.generate_scene(scene_id, n_obj, seed=seed + i)
            annotations.append(ann)
            print(f"  [{i+1}/{n_scenes}] {scene_id}: {len(ann.objects)} objects, env={ann.environment}")

        # Save manifest
        manifest_path = str(self.output_dir / "annotations.json")
        with open(manifest_path, "w") as f:
            json.dump([asdict(a) for a in annotations], f, indent=2)
        print(f"[Replicator] Dataset saved: {manifest_path}")
        return annotations
