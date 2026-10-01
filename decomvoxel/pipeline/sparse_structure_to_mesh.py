"""
Sparse Structure to Mesh Pipeline

Converts a completed sparse structure (64^3 occupancy grid) to a textured mesh using TRELLIS's structured latent flow model and mesh decoder.

Pipeline:
    1. Image conditioning: Encode input image via DINOv2 -> patch tokens
    2. Sparse structure -> coordinates: Extract occupied voxel positions
    3. Structured Latent Sampling: SLatFlowModel (Sparse Flow Transformer)
       generates structured latents conditioned on image + sparse coords
    4. Mesh Decoding: SLatMeshDecoder decodes structured latents -> mesh
    5. (Optional) Post-processing: simplify, texture bake, export to GLB/OBJ

"""

import os
import sys
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)


@dataclass
class SSToMeshConfig:
    """Configuration for sparse structure to mesh conversion."""

    # Structured latent sampling
    slat_sampler_steps: int = 12
    slat_sampler_cfg_strength: float = 3.0

    # Mesh export
    simplify_ratio: float = 0.95
    texture_size: int = 1024
    fill_holes: bool = True

    # Random seed
    seed: int = 42

    # Device
    device: str = "cuda"


class TrellisModelManager:
    """
    Manages loading and caching of TRELLIS pretrained models.
    Loads the full TrellisImageTo3DPipeline which contains:
        - image_cond_model (DINOv2): for image encoding
        - slat_flow_model (SLatFlowModel): structured latent flow transformer
        - slat_decoder_mesh (SLatMeshDecoder): structured latent -> mesh
        - slat_decoder_gs (SLatGaussianDecoder): structured latent -> gaussian
        - slat_sampler: flow euler sampler for structured latents
        - slat_normalization: mean/std for structured latent denormalization
    """

    _instance = None
    _pipeline = None

    def __new__(cls):
        """Singleton pattern to avoid loading models multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, device: str = "cuda"):
        """Load the TRELLIS pipeline if not already loaded."""
        if self._pipeline is not None:
            return

        from trellis.pipelines import TrellisImageTo3DPipeline
        print("[TrellisModelManager] Loading TRELLIS pipeline from HuggingFace...")
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        self._pipeline.to(torch.device(device))
        print("[TrellisModelManager] Pipeline loaded successfully.")

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._pipeline

    def get_image_encoder(self):
        return self.pipeline.models['image_cond_model']

    def get_slat_flow_model(self):
        return self.pipeline.models['slat_flow_model']

    def get_slat_decoder_mesh(self):
        return self.pipeline.models['slat_decoder_mesh']

    def get_slat_decoder_gs(self):
        return self.pipeline.models['slat_decoder_gs']

    def get_slat_sampler(self):
        return self.pipeline.slat_sampler

    def get_slat_sampler_params(self):
        return self.pipeline.slat_sampler_params

    def get_slat_normalization(self):
        return self.pipeline.slat_normalization


class ImageConditioner:
    """
    Encodes an input image into conditioning features using DINOv2.
    Wraps the pipeline's preprocess_image + encode_image + get_cond logic.
    """

    def __init__(self, model_manager: TrellisModelManager):
        self.manager = model_manager

    @torch.no_grad()
    def prepare(self, image: Union[str, Image.Image],
                preprocess: bool = True) -> Dict[str, torch.Tensor]:
        """
        Prepare conditioning from an image.

        Args:
            image: path to image file or PIL Image
            preprocess: whether to run TRELLIS preprocessing (background removal, resize)

        Returns:
            dict with 'cond' and 'neg_cond' tensors
        """
        pipe = self.manager.pipeline

        if isinstance(image, str):
            image = Image.open(image).convert('RGBA')

        if preprocess:
            image = pipe.preprocess_image(image)

        cond = pipe.get_cond([image])
        return cond


class StructuredLatentSampler:
    """
    Samples structured latents from sparse structure coordinates + image condition
    using the SLatFlowModel (Sparse Flow Transformer) with flow-matching.
    """

    def __init__(self, model_manager: TrellisModelManager):
        self.manager = model_manager

    @torch.no_grad()
    def sample(self, coords: torch.Tensor, cond: Dict[str, torch.Tensor],
               sampler_params: Optional[dict] = None):
        """
        Sample structured latents for the given sparse coordinates.

        Args:
            coords: (N, 4) int tensor — [batch_idx, x, y, z] for occupied voxels
            cond: conditioning dict with 'cond' and 'neg_cond'
            sampler_params: override sampler parameters (steps, cfg_strength, etc.)

        Returns:
            sp.SparseTensor: the sampled structured latent
        """
        from trellis.modules import sparse as sp

        flow_model = self.manager.get_slat_flow_model()
        sampler = self.manager.get_slat_sampler()
        default_params = self.manager.get_slat_sampler_params()
        normalization = self.manager.get_slat_normalization()

        # Merge parameters
        merged_params = {**default_params}
        if sampler_params:
            merged_params.update(sampler_params)

        device = next(flow_model.parameters()).device

        # Build noise SparseTensor on the given coordinates
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(device),
            coords=coords.to(device),
        )

        # Run flow-matching sampling
        slat = sampler.sample(
            flow_model,
            noise,
            **cond,
            **merged_params,
            verbose=True
        )
        slat_samples = slat.samples

        # Denormalize structured latent
        std = torch.tensor(normalization['std'])[None].to(slat_samples.device)
        mean = torch.tensor(normalization['mean'])[None].to(slat_samples.device)
        slat_samples = slat_samples * std + mean

        return slat_samples


class MeshDecoder:
    """
    Decodes structured latents into mesh using SLatMeshDecoder.
    Optionally also decodes Gaussian representation for texture baking.
    """

    def __init__(self, model_manager: TrellisModelManager):
        self.manager = model_manager

    @torch.no_grad()
    def decode(self, slat, formats: List[str] = None):
        """
        Decode structured latent into output representations.

        Args:
            slat: SparseTensor structured latent
            formats: list of output formats, subset of ['mesh', 'gaussian', 'radiance_field']
                     defaults to ['mesh', 'gaussian']

        Returns:
            dict mapping format name -> decoded result
        """
        if formats is None:
            formats = ['mesh', 'gaussian']

        pipe = self.manager.pipeline
        return pipe.decode_slat(slat, formats)


class MeshExporter:
    """
    Exports MeshExtractResult to various file formats (GLB, OBJ, PLY).
    Supports texture baking from Gaussian representation.
    """

    @staticmethod
    def export_glb(mesh_result, gaussian_result=None,
                   output_path: str = "output.glb",
                   simplify: float = 0.95,
                   texture_size: int = 1024,
                   fill_holes: bool = True) -> str:
        """
        Export mesh to GLB format with baked texture.

        Args:
            mesh_result: MeshExtractResult from decoder
            gaussian_result: Gaussian representation for texture baking (optional)
            output_path: path to save GLB file
            simplify: face simplification ratio (0 = no simplification, 0.95 = remove 95%)
            texture_size: texture resolution
            fill_holes: whether to fill holes in mesh

        Returns:
            path to saved GLB file
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if gaussian_result is not None:
            from trellis.utils.postprocessing_utils import to_glb
            with torch.enable_grad():
                trimesh_mesh = to_glb(
                    app_rep=gaussian_result,
                    mesh=mesh_result,
                    simplify=simplify,
                    fill_holes=fill_holes,
                    texture_size=texture_size,
                    verbose=True
                )
            trimesh_mesh.export(output_path)
        else:
            # Export without texture
            import trimesh
            vertices = mesh_result.vertices.detach().cpu().numpy()
            faces = mesh_result.faces.detach().cpu().numpy()

            # Rotate from z-up to y-up
            vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

            # Use vertex colors if available
            if mesh_result.vertex_attrs is not None:
                colors = mesh_result.vertex_attrs.detach().cpu().numpy()
                # Take RGB channels (first 3)
                if colors.shape[-1] >= 3:
                    colors = colors[:, :3]
                    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                           vertex_colors=colors, process=False)
                else:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            else:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.export(output_path)

        print(f"[MeshExporter] Saved GLB to: {output_path}")
        return output_path

    @staticmethod
    def export_obj(mesh_result, output_path: str = "output.obj") -> str:
        """
        Export mesh to OBJ format (geometry only, no texture).

        Args:
            mesh_result: MeshExtractResult from decoder
            output_path: path to save OBJ file

        Returns:
            path to saved OBJ file
        """
        import trimesh
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        vertices = mesh_result.vertices.detach().cpu().numpy()
        faces = mesh_result.faces.detach().cpu().numpy()
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(output_path)
        print(f"[MeshExporter] Saved OBJ to: {output_path}")
        return output_path

    @staticmethod
    def export_ply(mesh_result, output_path: str = "output.ply") -> str:
        """
        Export mesh to PLY format.

        Args:
            mesh_result: MeshExtractResult from decoder
            output_path: path to save PLY file

        Returns:
            path to saved PLY file
        """
        import trimesh
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        vertices = mesh_result.vertices.detach().cpu().numpy()
        faces = mesh_result.faces.detach().cpu().numpy()
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        if mesh_result.vertex_attrs is not None:
            colors = mesh_result.vertex_attrs.detach().cpu().numpy()
            if colors.shape[-1] >= 3:
                colors = colors[:, :3]
                colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                                       vertex_colors=colors, process=False)
            else:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        mesh.export(output_path)
        print(f"[MeshExporter] Saved PLY to: {output_path}")
        return output_path


class SparseStructureToMeshPipeline:
    """
    End-to-end pipeline: sparse structure (64^3) + image -> mesh.

    Usage:
        pipeline = SparseStructureToMeshPipeline(cfg)
        pipeline.initialize()
        result = pipeline.run(sparse_structure, image_path, output_dir)
    """

    def __init__(self, cfg: Optional[SSToMeshConfig] = None):
        self.cfg = cfg or SSToMeshConfig()
        self.model_manager = TrellisModelManager()
        self.conditioner = ImageConditioner(self.model_manager)
        self.latent_sampler = StructuredLatentSampler(self.model_manager)
        self.mesh_decoder = MeshDecoder(self.model_manager)
        self.mesh_exporter = MeshExporter()
        self._initialized = False

    def initialize(self):
        """Load all TRELLIS pretrained models."""
        if self._initialized:
            return
        self.model_manager.load(device=self.cfg.device)
        self._initialized = True

    @staticmethod
    def sparse_structure_to_coords(sparse_structure: torch.Tensor,
                                   threshold: float = 0.5) -> torch.Tensor:
        """
        Convert a (1, 1, 64, 64, 64) sparse structure to coordinate tensor.

        Args:
            sparse_structure: occupancy grid (1, 1, D, H, W)
            threshold: binarization threshold

        Returns:
            (N, 4) int tensor with columns [batch_idx, x, y, z]
        """
        coords = torch.argwhere(sparse_structure > threshold)  # (N, 5): [b, c, x, y, z]
        coords = coords[:, [0, 2, 3, 4]].int()  # -> [b, x, y, z]
        return coords

    @torch.no_grad()
    def run(self,
            sparse_structure: torch.Tensor,
            image: Union[str, Image.Image],
            output_dir: str,
            formats: List[str] = None,
            threshold: float = 0.5,
            preprocess_image: bool = True,
            export_glb: bool = True,
            export_obj: bool = False,
            export_ply: bool = False,
            slat_sampler_params: Optional[dict] = None,
            ) -> Dict[str, Any]:
        """
        Run the full sparse structure -> mesh pipeline.

        Args:
            sparse_structure: (1, 1, 64, 64, 64) occupancy grid
            image: conditioning image (path or PIL Image)
            output_dir: directory to save outputs
            formats: decoder output formats (default: ['mesh', 'gaussian'])
            threshold: occupancy threshold for coordinate extraction
            preprocess_image: whether to preprocess image (background removal)
            export_glb: whether to export GLB file
            export_obj: whether to export OBJ file
            export_ply: whether to export PLY file
            slat_sampler_params: override structured latent sampler parameters

        Returns:
            dict with keys:
                'mesh': list of MeshExtractResult
                'gaussian': list of Gaussian (if requested)
                'coords': coordinate tensor
                'cond': conditioning dict
                'slat': structured latent SparseTensor
                'exported_files': dict of exported file paths
        """
        self.initialize()

        if formats is None:
            formats = ['mesh', 'gaussian']

        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Prepare image conditioning
        print("[SS2Mesh] Step 1: Encoding conditioning image...")
        cond = self.conditioner.prepare(image, preprocess=preprocess_image)

        # Step 2: Extract coordinates from sparse structure
        print("[SS2Mesh] Step 2: Extracting coordinates from sparse structure...")
        device = torch.device(self.cfg.device)
        if sparse_structure.device != device:
            sparse_structure = sparse_structure.to(device)
        coords = self.sparse_structure_to_coords(sparse_structure, threshold=threshold)
        num_occupied = coords.shape[0]
        print(f"[SS2Mesh] Found {num_occupied} occupied voxels")

        if num_occupied == 0:
            print("[SS2Mesh] WARNING: No occupied voxels found. Cannot generate mesh.")
            return {'mesh': None, 'coords': coords, 'exported_files': {}}

        # Step 3: Sample structured latents
        print("[SS2Mesh] Step 3: Sampling structured latents via flow transformer...")
        torch.manual_seed(self.cfg.seed)

        merged_sampler_params = {
            'steps': self.cfg.slat_sampler_steps,
            'cfg_strength': self.cfg.slat_sampler_cfg_strength,
        }
        if slat_sampler_params:
            merged_sampler_params.update(slat_sampler_params)

        slat = self.latent_sampler.sample(coords, cond, sampler_params=merged_sampler_params)

        # Step 4: Decode structured latent to mesh
        print("[SS2Mesh] Step 4: Decoding structured latent to mesh...")
        try:
            decoded = self.mesh_decoder.decode(slat, formats=formats)
        except:
            print(f"[SS2Mesh] ERROR during decoding: product overflows int32.")
            return {'mesh': None, 'coords': coords, 'exported_files': {}}

        # Step 5: Export mesh files
        print("[SS2Mesh] Step 5: Exporting mesh...")
        exported_files = {}

        mesh_results = decoded.get('mesh', [])
        gaussian_results = decoded.get('gaussian', [])

        if mesh_results and len(mesh_results) > 0:
            mesh_result = mesh_results[0]

            if mesh_result is not None and mesh_result.success:
                gaussian_result = gaussian_results[0] if gaussian_results else None

                if export_glb:
                    glb_path = os.path.join(output_dir, 'output.glb')
                    self.mesh_exporter.export_glb(
                        mesh_result=mesh_result,
                        gaussian_result=gaussian_result,
                        output_path=glb_path,
                        simplify=self.cfg.simplify_ratio,
                        texture_size=self.cfg.texture_size,
                        fill_holes=self.cfg.fill_holes
                    )
                    exported_files['glb'] = glb_path

                if export_obj:
                    obj_path = os.path.join(output_dir, 'output.obj')
                    self.mesh_exporter.export_obj(mesh_result, output_path=obj_path)
                    exported_files['obj'] = obj_path

                if export_ply:
                    ply_path = os.path.join(output_dir, 'output.ply')
                    self.mesh_exporter.export_ply(mesh_result, output_path=ply_path)
                    exported_files['ply'] = ply_path

                print(f"[SS2Mesh] Mesh: {mesh_result.vertices.shape[0]} vertices, "
                      f"{mesh_result.faces.shape[0]} faces")
            else:
                print("[SS2Mesh] WARNING: Mesh extraction failed (empty mesh).")
        else:
            print("[SS2Mesh] WARNING: No mesh decoded.")

        print("[SS2Mesh] Pipeline complete.")
        return {
            'mesh': mesh_results,
            'gaussian': gaussian_results,
            'coords': coords,
            'cond': cond,
            'slat': slat,
            'exported_files': exported_files,
        }


# ============================================================
# Convenience functions for external use
# ============================================================

def sparse_structure_to_mesh(
    sparse_structure: torch.Tensor,
    image: Union[str, Image.Image],
    output_dir: str,
    cfg: Optional[SSToMeshConfig] = None,
    threshold: float = 0.5,
    preprocess_image: bool = True,
    export_glb: bool = True,
    export_obj: bool = False,
    export_ply: bool = False,
    slat_sampler_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert a sparse structure occupancy grid to mesh, conditioned on an image.

    This is the main entry point for the sparse structure -> mesh conversion.

    Args:
        sparse_structure: (1, 1, 64, 64, 64) occupancy grid tensor
        image: conditioning image (file path or PIL Image)
        output_dir: directory to save output mesh files
        cfg: pipeline configuration (SSToMeshConfig)
        threshold: occupancy threshold for binarizing the sparse structure
        preprocess_image: whether to preprocess image (background removal, resize)
        export_glb: export GLB file with baked texture
        export_obj: export OBJ file (geometry only)
        export_ply: export PLY file
        slat_sampler_params: override sampler params (e.g., {'steps': 20, 'cfg_strength': 5.0})

    Returns:
        dict containing mesh results, coordinates, exported file paths, etc.
    """
    pipeline = SparseStructureToMeshPipeline(cfg=cfg)
    return pipeline.run(
        sparse_structure=sparse_structure,
        image=image,
        output_dir=output_dir,
        threshold=threshold,
        preprocess_image=preprocess_image,
        export_glb=export_glb,
        export_obj=export_obj,
        export_ply=export_ply,
        slat_sampler_params=slat_sampler_params,
    )


def coords_to_mesh(
    coords: torch.Tensor,
    image: Union[str, Image.Image],
    output_dir: str,
    cfg: Optional[SSToMeshConfig] = None,
    preprocess_image: bool = True,
    export_glb: bool = True,
    slat_sampler_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert pre-extracted coordinates to mesh, conditioned on an image.

    Useful when you already have the voxel coordinates and want to skip
    the sparse structure -> coords step.

    Args:
        coords: (N, 4) int tensor with columns [batch_idx, x, y, z]
        image: conditioning image (file path or PIL Image)
        output_dir: directory to save outputs
        cfg: pipeline configuration
        preprocess_image: whether to preprocess image
        export_glb: export GLB file
        slat_sampler_params: override sampler params

    Returns:
        dict with mesh results and exported files
    """
    cfg = cfg or SSToMeshConfig()
    pipeline = SparseStructureToMeshPipeline(cfg=cfg)
    pipeline.initialize()

    os.makedirs(output_dir, exist_ok=True)

    # Prepare conditioning
    cond = pipeline.conditioner.prepare(image, preprocess=preprocess_image)

    # Sample structured latent
    torch.manual_seed(cfg.seed)
    merged_params = {'steps': cfg.slat_sampler_steps, 'cfg_strength': cfg.slat_sampler_cfg_strength}
    if slat_sampler_params:
        merged_params.update(slat_sampler_params)
    slat = pipeline.latent_sampler.sample(coords, cond, sampler_params=merged_params)

    # Decode
    decoded = pipeline.mesh_decoder.decode(slat, formats=['mesh', 'gaussian'])

    # Export
    exported_files = {}
    mesh_results = decoded.get('mesh', [])
    gaussian_results = decoded.get('gaussian', [])
    if mesh_results and mesh_results[0] is not None and mesh_results[0].success:
        if export_glb:
            glb_path = os.path.join(output_dir, 'output.glb')
            gaussian_result = gaussian_results[0] if gaussian_results else None
            MeshExporter.export_glb(mesh_results[0], gaussian_result, glb_path,
                                    simplify=cfg.simplify_ratio,
                                    texture_size=cfg.texture_size,
                                    fill_holes=cfg.fill_holes)
            exported_files['glb'] = glb_path

    return {
        'mesh': mesh_results,
        'gaussian': gaussian_results,
        'coords': coords,
        'slat': slat,
        'exported_files': exported_files,
    }


# ============================================================
# Test section
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sparse Structure to Mesh Pipeline")
    parser.add_argument('--ss_path', type=str, required=True,
                        help='Path to sparse structure .pt file (1,1,64,64,64)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to conditioning image')
    parser.add_argument('--output_dir', type=str, default='outputs/ss_to_mesh',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--steps', type=int, default=12,
                        help='Number of sampler steps')
    parser.add_argument('--cfg_strength', type=float, default=3.0,
                        help='CFG strength for structured latent sampling')
    parser.add_argument('--simplify', type=float, default=0.95,
                        help='Mesh simplification ratio')
    parser.add_argument('--texture_size', type=int, default=1024,
                        help='Texture resolution')
    parser.add_argument('--export_obj', action='store_true', default=False,
                        help='Also export OBJ')
    parser.add_argument('--export_ply', action='store_true', default=False,
                        help='Also export PLY')
    parser.add_argument('--no_preprocess', action='store_true', default=False,
                        help='Skip image preprocessing')
    args = parser.parse_args()

    # Load sparse structure
    print(f"Loading sparse structure from: {args.ss_path}")
    ss = torch.load(args.ss_path, map_location='cpu')
    if ss.dim() == 3:
        ss = ss.unsqueeze(0).unsqueeze(0)
    elif ss.dim() == 4:
        ss = ss.unsqueeze(0)
    print(f"Sparse structure shape: {ss.shape}")
    print(f"Occupied voxels (>{args.threshold}): {(ss > args.threshold).sum().item()}")

    # Configure
    cfg = SSToMeshConfig(
        seed=args.seed,
        slat_sampler_steps=args.steps,
        slat_sampler_cfg_strength=args.cfg_strength,
        simplify_ratio=args.simplify,
        texture_size=args.texture_size,
    )

    # Run pipeline
    result = sparse_structure_to_mesh(
        sparse_structure=ss,
        image=args.image,
        output_dir=args.output_dir,
        cfg=cfg,
        threshold=args.threshold,
        preprocess_image=not args.no_preprocess,
        export_glb=True,
        export_obj=args.export_obj,
        export_ply=args.export_ply,
    )

    print("\n" + "=" * 60)
    print("Results:")
    for key, path in result.get('exported_files', {}).items():
        print(f"  {key}: {path}")
    print("=" * 60)
