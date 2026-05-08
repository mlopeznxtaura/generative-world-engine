"""
Depth Pro monocular depth estimation.
Estimate metric depth from a single RGB image — no stereo, no LiDAR.
SDKs: Depth Pro, PyTorch, OpenCV, Pillow
"""
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from PIL import Image

try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    print("Warning: Depth Pro not available. Install from: https://github.com/apple/ml-depth-pro")


class DepthEstimator:
    """
    Monocular metric depth estimation using Apple Depth Pro.
    Returns metric depth maps aligned with input images.
    Falls back to MiDaS (relative depth) when Depth Pro is unavailable.
    """

    def __init__(self, device: str = "cuda", use_midas_fallback: bool = True):
        self.device = device
        self.model = None
        self.transform = None
        self.use_midas = False

        if DEPTH_PRO_AVAILABLE:
            print("[DepthPro] Loading model...")
            self.model, self.transform = depth_pro.create_model_and_transforms(device=device)
            self.model.eval()
            print("[DepthPro] Ready")
        elif use_midas_fallback:
            print("[DepthPro] Falling back to MiDaS relative depth...")
            self._load_midas(device)
        else:
            print("[DepthPro] No depth model loaded.")

    def _load_midas(self, device: str):
        """Load MiDaS as a fallback relative depth estimator."""
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self.model.to(device).eval()
            self.use_midas = True
            print("[MiDaS] Loaded small model")
        except Exception as e:
            print(f"[MiDaS] Failed to load: {e}")

    @torch.no_grad()
    def estimate(
        self,
        image: Union[str, np.ndarray, Image.Image],
        output_path: Optional[str] = None,
        colorize: bool = True,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Estimate depth from a single image.
        Returns dict with: depth (H,W float32), focal_length, colorized_depth
        """
        # Load image
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image, np.ndarray):
            np_img = image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
        else:
            pil_img = image
            np_img = np.array(pil_img)

        if DEPTH_PRO_AVAILABLE and not self.use_midas:
            result = self._estimate_depth_pro(pil_img)
        elif self.use_midas and self.model is not None:
            result = self._estimate_midas(np_img)
        else:
            # Stub: return gradient depth
            h, w = np_img.shape[:2]
            depth = np.linspace(1.0, 10.0, h * w).reshape(h, w).astype(np.float32)
            result = {"depth": depth, "focal_length": 500.0, "metric": False}

        if colorize:
            depth_norm = (result["depth"] - result["depth"].min())
            depth_norm = depth_norm / (depth_norm.max() + 1e-8)
            colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            result["colorized"] = colored

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if "colorized" in result:
                cv2.imwrite(output_path, result["colorized"])
            else:
                depth_vis = (result["depth"] / result["depth"].max() * 255).astype(np.uint8)
                cv2.imwrite(output_path, depth_vis)
            print(f"[Depth] Saved: {output_path}")

        return result

    def _estimate_depth_pro(self, pil_img: Image.Image) -> Dict:
        prediction = self.model.infer(self.transform(pil_img))
        depth = prediction["depth"].cpu().numpy()
        focal = float(prediction.get("focallength_px", 500.0))
        return {"depth": depth, "focal_length": focal, "metric": True}

    def _estimate_midas(self, np_img: np.ndarray) -> Dict:
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) if np_img.ndim == 3 else np_img
        inp = self.transform(img_rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=np_img.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()
        return {"depth": pred.astype(np.float32), "focal_length": None, "metric": False}

    def estimate_batch(
        self, image_paths: list, output_dir: str = "./depths"
    ) -> list:
        """Estimate depth for a batch of images."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for i, path in enumerate(image_paths):
            stem = Path(path).stem
            out = str(Path(output_dir) / f"{stem}_depth.png")
            result = self.estimate(path, output_path=out)
            results.append({"input": path, "output": out, "metric": result.get("metric", False)})
            print(f"[Depth] {i+1}/{len(image_paths)}: {stem}")
        return results
