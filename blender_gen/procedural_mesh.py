"""
Blender Python API procedural mesh generation.
Generate terrain, buildings, vegetation, and props via bpy scripts.
SDKs: Blender Python API (bpy), NumPy
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# The Blender script runs inside a Blender subprocess (not the host Python).
# bpy is only importable inside a running Blender process.
_BLENDER_SCRIPT = """\
import bpy
import numpy as np
import json
import sys
import math

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def add_terrain(size=20.0, subdivisions=32, height_scale=2.0, seed=42):
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.active_object
    plane.name = "Terrain"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=subdivisions)
    bpy.ops.object.mode_set(mode="OBJECT")
    rng = np.random.default_rng(seed)
    mesh = plane.data
    for v in mesh.vertices:
        v.co.z = float(rng.uniform(0, 1)) * height_scale * math.exp(
            -0.1 * (v.co.x**2 + v.co.y**2) / (size / 4)**2
        )
    mesh.update()
    bpy.ops.object.shade_smooth()
    return plane

def add_cube(name, location, size=1.0):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj

def add_sphere(name, location, radius=0.5):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=location, segments=32, ring_count=16
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj

def add_camera(location, look_at=(0, 0, 0)):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    return cam

def add_sun_light(energy=5.0):
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = energy
    return sun

def export_glb(output_path):
    bpy.ops.export_scene.gltf(
        filepath=output_path, export_format="GLB",
        export_apply=True, export_materials="EXPORT",
    )

config = json.loads(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else {}
scene_type = config.get("scene_type", "terrain")
output_path = config.get("output_path", "/tmp/scene.glb")
seed = config.get("seed", 42)
n_objects = config.get("n_objects", 10)

clear_scene()
add_sun_light()
add_camera(location=(10, -10, 7))

if scene_type == "terrain":
    add_terrain(size=20.0, subdivisions=32, height_scale=2.0, seed=seed)
elif scene_type == "objects":
    rng = np.random.default_rng(seed)
    for i in range(n_objects):
        pos = rng.uniform(-5, 5, 3).tolist()
        pos[2] = abs(pos[2])
        if rng.random() > 0.5:
            add_cube(f"Cube_{i:03d}", pos, size=float(rng.uniform(0.5, 2.0)))
        else:
            add_sphere(f"Sphere_{i:03d}", pos, radius=float(rng.uniform(0.3, 1.0)))

export_glb(output_path)
print(f"Exported: {output_path}")
"""


class BlenderMeshGenerator:
    """
    Drive Blender as a subprocess for headless procedural mesh generation.
    Requires Blender installed at BLENDER_PATH env var or system PATH.

    Usage:
        gen = BlenderMeshGenerator()
        gen.generate(scene_type="terrain", output_path="./terrain.glb")
        gen.generate_dataset(n_scenes=10, output_dir="./scenes")
    """

    def __init__(self, blender_path: Optional[str] = None):
        self.blender_path = blender_path or os.environ.get("BLENDER_PATH", "blender")
        self._check_blender()

    def _check_blender(self):
        try:
            result = subprocess.run(
                [self.blender_path, "--version"],
                capture_output=True, text=True, timeout=10
            )
            print(f"[Blender] {result.stdout.split(chr(10))[0]}")
        except FileNotFoundError:
            print(f"[Blender] Not found at '{self.blender_path}'. Set BLENDER_PATH env var.")
        except subprocess.TimeoutExpired:
            print("[Blender] Version check timed out.")

    def generate(
        self,
        scene_type: str = "terrain",
        output_path: str = "/tmp/scene.glb",
        n_objects: int = 10,
        seed: int = 42,
    ) -> str:
        """Generate a scene using headless Blender and export as GLB."""
        import tempfile
        config = json.dumps({
            "scene_type": scene_type,
            "output_path": str(output_path),
            "n_objects": n_objects,
            "seed": seed,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(_BLENDER_SCRIPT)
            script_path = f.name

        cmd = [self.blender_path, "--background", "--python", script_path, "--", config]
        print(f"[Blender] Generating {scene_type} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        os.unlink(script_path)

        if result.returncode != 0:
            print(f"[Blender] Error: {result.stderr[-400:]}")
        else:
            print(f"[Blender] Done: {output_path}")
        return output_path

    def generate_dataset(
        self,
        n_scenes: int = 10,
        scene_type: str = "terrain",
        output_dir: str = "./blender_scenes",
        seed: int = 42,
    ) -> List[str]:
        """Generate N scenes, return list of GLB paths."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n_scenes):
            out = str(Path(output_dir) / f"scene_{i:04d}.glb")
            self.generate(scene_type=scene_type, output_path=out, seed=seed + i)
            paths.append(out)
        print(f"[Blender] Generated {n_scenes} {scene_type} scenes -> {output_dir}")
        return paths
