"""
OpenUSD scene construction and asset management.
Build structured USD stages with prims, materials, lights, and cameras.
SDKs: OpenUSD (pxr), NVIDIA Omniverse
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf, Vt
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    print("Warning: OpenUSD (pxr) not available. Install: pip install usd-core")


class USDSceneBuilder:
    """
    Build and modify USD stages programmatically.
    Create scenes, add assets, configure materials and lighting.
    """

    def __init__(self, stage_path: str = "/tmp/scene.usda"):
        if not USD_AVAILABLE:
            raise ImportError("OpenUSD required. Install: pip install usd-core")
        self.stage_path = stage_path
        self.stage = Usd.Stage.CreateNew(stage_path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        # Root xform
        self.root = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(self.root.GetPrim())
        print(f"[USD] Stage created: {stage_path}")

    def add_ground_plane(self, size: float = 20.0, y: float = 0.0) -> "UsdGeom.Mesh":
        """Add a flat ground plane mesh."""
        plane = UsdGeom.Mesh.Define(self.stage, "/World/GroundPlane")
        half = size / 2.0
        points = [
            Gf.Vec3f(-half, y, -half), Gf.Vec3f(half, y, -half),
            Gf.Vec3f(half, y, half),  Gf.Vec3f(-half, y, half),
        ]
        plane.GetPointsAttr().Set(Vt.Vec3fArray(points))
        plane.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        plane.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        return plane

    def add_cube(
        self, name: str, position: Tuple[float, float, float] = (0, 0, 0),
        size: float = 1.0, color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> "UsdGeom.Cube":
        """Add a cube prim at given position."""
        path = f"/World/{name}"
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.GetSizeAttr().Set(size)

        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*position))

        # Material
        mat_path = f"/World/Materials/{name}_mat"
        material = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI(cube).Bind(material)
        return cube

    def add_sphere(
        self, name: str, position: Tuple[float, float, float] = (0, 1, 0),
        radius: float = 0.5, color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    ) -> "UsdGeom.Sphere":
        sphere = UsdGeom.Sphere.Define(self.stage, f"/World/{name}")
        sphere.GetRadiusAttr().Set(radius)
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*position))
        return sphere

    def add_distant_light(
        self, name: str = "SunLight",
        intensity: float = 1000.0,
        angle: float = 0.53,
        color: Tuple[float, float, float] = (1.0, 0.98, 0.92),
        rotation: Tuple[float, float, float] = (-45, 30, 0),
    ) -> "UsdLux.DistantLight":
        light = UsdLux.DistantLight.Define(self.stage, f"/World/{name}")
        light.GetIntensityAttr().Set(intensity)
        light.GetAngleAttr().Set(angle)
        light.GetColorAttr().Set(Gf.Vec3f(*color))
        UsdGeom.XformCommonAPI(light).SetRotate(Gf.Vec3f(*rotation))
        return light

    def add_dome_light(
        self, name: str = "DomeLight",
        intensity: float = 500.0,
        texture_path: Optional[str] = None,
    ) -> "UsdLux.DomeLight":
        light = UsdLux.DomeLight.Define(self.stage, f"/World/{name}")
        light.GetIntensityAttr().Set(intensity)
        if texture_path:
            light.GetTextureFileAttr().Set(texture_path)
        return light

    def add_camera(
        self, name: str = "Camera",
        position: Tuple[float, float, float] = (0, 2, 5),
        look_at: Tuple[float, float, float] = (0, 0, 0),
        focal_length: float = 35.0,
        h_aperture: float = 36.0,
    ) -> "UsdGeom.Camera":
        cam = UsdGeom.Camera.Define(self.stage, f"/World/{name}")
        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetHorizontalApertureAttr().Set(h_aperture)

        # Point camera at target
        pos = np.array(position, dtype=np.float64)
        target = np.array(look_at, dtype=np.float64)
        forward = target - pos
        forward /= np.linalg.norm(forward)

        xform = UsdGeom.XformCommonAPI(cam)
        xform.SetTranslate(Gf.Vec3d(*position))
        return cam

    def populate_random_scene(
        self, n_objects: int = 10, world_size: float = 8.0, seed: int = 42
    ):
        """Populate stage with random objects, lighting, and a camera."""
        rng = np.random.default_rng(seed)
        self.add_ground_plane(size=world_size * 2)
        self.add_distant_light()
        self.add_dome_light()
        self.add_camera(position=(8, 5, 8))

        shapes = ["cube", "sphere"]
        for i in range(n_objects):
            pos = (
                float(rng.uniform(-world_size/2, world_size/2)),
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(-world_size/2, world_size/2)),
            )
            color = tuple(float(c) for c in rng.uniform(0.1, 0.9, 3))
            name = f"Object_{i:03d}"
            if rng.random() > 0.5:
                self.add_cube(name, pos, float(rng.uniform(0.3, 1.5)), color)
            else:
                self.add_sphere(name, pos, float(rng.uniform(0.2, 0.8)), color)

    def save(self) -> str:
        self.stage.Save()
        print(f"[USD] Stage saved: {self.stage_path}")
        return self.stage_path

    def export_usdc(self, output_path: str) -> str:
        """Export as binary USDC (smaller, faster to load)."""
        self.stage.Export(output_path)
        print(f"[USD] Exported USDC: {output_path}")
        return output_path
