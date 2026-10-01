"""
Score Distillation Sampling (SDS) for Voxel Completion

This module provides SDS-based voxel completion using TRELLIS diffusion model.
It takes a partial/incomplete voxel representation and completes it using
score distillation from a pre-trained 3D diffusion model.

The SDS process follows these steps:
1. Current estimate of complete voxels: Z_0'
2. Add noise epsilon: Z_1' = Z_0' + epsilon
3. Use incomplete voxel Z_2 to guide: constrain known regions on Z_1'
4. Flow Transformer predicts noise: epsilon_hat = model(Z_1', t, condition)
5. Compute gradient: nabla proportional to (epsilon_hat - epsilon)
6. Update Z_0': move along gradient direction

Key Components:
1. SDSConfig: Configuration for SDS optimization
2. SDSLoss: Computes score distillation sampling loss
3. TRELLISWrapper: Wrapper for TRELLIS models
4. VoxelCompleter: Main class for running voxel completion
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
from PIL import Image
from dataclasses import dataclass

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDSConfig:
    """
    Configuration for Score Distillation Sampling.
    
    Attributes:
        total_iters: number of optimization iterations
        lr: learning rate for HashGrid optimization
        noise_start: minimum noise level for sampling
        noise_end: maximum noise level for sampling
        sds_weight: weight for SDS latent loss
        voxel_weight: weight for voxel reconstruction loss
        mask_weight: weight for known region preservation loss
        cfg_strength: classifier-free guidance strength
        sample_coeff: sampling coefficient for anti-aliasing
        sample_interval: sampling interval for neighborhood sampling
        refine_noise: noise level for final refinement
        refine_steps: number of denoising steps for refinement
        print_interval: interval for printing progress
        save_interval: interval for saving intermediate results
    """
    
    def __init__(self,
                 total_iters: int = 5000,
                 lr: float = 0.001,
                 noise_start: float = 0.02,
                 noise_end: float = 0.98,
                 sds_weight: float = 1.0,
                 voxel_weight: float = 0.1,
                 mask_weight: float = 0.5,
                 cfg_strength: float = 3.0,
                 sample_coeff: float = 1.0,
                 sample_interval: float = 0.5,
                 refine_noise: float = 0.3,
                 refine_steps: int = 25,
                 print_interval: int = 100,
                 save_interval: int = 500):
        self.total_iters = total_iters
        self.lr = lr
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.sds_weight = sds_weight
        self.voxel_weight = voxel_weight
        self.mask_weight = mask_weight
        self.cfg_strength = cfg_strength
        self.sample_coeff = sample_coeff
        self.sample_interval = sample_interval
        self.refine_noise = refine_noise
        self.refine_steps = refine_steps
        self.print_interval = print_interval
        self.save_interval = save_interval


class SDSLoss(nn.Module):
    """
    Score Distillation Sampling Loss with Known Region Preservation.
    
    This computes the SDS loss by:
    1. Encoding the current voxel state to latent space
    2. Adding noise at a random timestep (flow matching: x_t = (1-t)*x + t*noise)
    3. Predicting the denoised state using the diffusion model
    4. Computing the gradient direction for optimization
    5. Optionally preserving known regions using a binary mask
    
    The flow matching formulation follows TRELLIS:
    - Forward: x_t = (1-t)*x + t*noise
    - Backward: predict x_0 from x_t
    """
    
    def __init__(self, encoder, decoder, diffusion, sampler, 
                 cfg: SDSConfig = None):
        """
        Initialize SDS loss.
        
        Args:
            encoder: voxel encoder model (sparse_structure_encoder)
            decoder: voxel decoder model (sparse_structure_decoder)
            diffusion: diffusion model (sparse_structure_flow_model)
            sampler: noise sampler (FlowMatchingEulerSampler)
            cfg: SDS configuration
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.sampler = sampler
        self.cfg = cfg or SDSConfig()
    
    def forward(self, voxels: torch.Tensor, 
                condition: Dict,
                known_mask: Optional[torch.Tensor] = None,
                initial_voxels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute SDS loss with optional known region preservation.
        
        Args:
            voxels: (B, 1, 64, 64, 64) current voxel occupancy
            condition: conditioning dict with 'cond' and 'neg_cond'
            known_mask: (B, 1, 64, 64, 64) binary mask of known regions (1=known)
            initial_voxels: (B, 1, 64, 64, 64) initial voxel values for known regions
            
        Returns:
            tuple of (loss, info_dict)
        """
        # Encode voxels to latent space
        latents = self.encoder(voxels, sample_posterior=False)  # (B, 8, 16, 16, 16)
        
        # Sample random timestep uniformly in [noise_start, noise_end]
        t = torch.rand(1, device=voxels.device)
        t = t * (self.cfg.noise_end - self.cfg.noise_start) + self.cfg.noise_start
        # Rescale for flow matching (TRELLIS uses this transformation)
        t_rescaled = 3 * t / (1 + 2 * t)
        t_broadcast = t_rescaled[:, None, None, None, None]
        
        # Flow matching: add noise via linear interpolation
        # x_t = (1 - t) * x + t * noise
        noise = torch.randn_like(latents)
        x_t = (1 - t_broadcast) * latents + t_broadcast * noise
        
        # Predict denoised state using classifier-free guidance
        with torch.no_grad():
            x_0_pred, noise_pred = self.sampler.sample_once_eps(
                self.diffusion, x_t, t_rescaled.squeeze(),
                **condition,
                cfg_strength=self.cfg.cfg_strength,
                cfg_interval=[0.5, 1.0]
            )
        
        # SDS loss: minimize difference between current latent and predicted clean
        # This encourages the latent to move toward the diffusion prior
        loss_latent = self.cfg.sds_weight * F.mse_loss(latents, x_0_pred.detach())
        loss_latent = torch.nan_to_num(loss_latent)
        
        # Decode predicted voxels for voxel-space loss
        with torch.no_grad():
            voxels_pred = self.decoder(x_0_pred)
            voxels_pred_activated = torch.sigmoid(voxels_pred)
        
        # Voxel reconstruction loss
        loss_voxel = self.cfg.voxel_weight * F.mse_loss(voxels, voxels_pred_activated)
        loss_voxel = torch.nan_to_num(loss_voxel)
        
        # Known region preservation loss
        loss_mask = torch.tensor(0.0, device=voxels.device)
        if known_mask is not None and initial_voxels is not None:
            # Force voxels in known regions to stay close to initial values
            known_diff = (voxels - initial_voxels) * known_mask
            loss_mask = self.cfg.mask_weight * F.mse_loss(
                known_diff, torch.zeros_like(known_diff)
            )
            loss_mask = torch.nan_to_num(loss_mask)
        
        total_loss = loss_latent + loss_voxel + loss_mask
        
        info = {
            'loss_latent': loss_latent.item(),
            'loss_voxel': loss_voxel.item(),
            'loss_mask': loss_mask.item() if isinstance(loss_mask, torch.Tensor) else loss_mask,
            'timestep': t.squeeze().item(),
            'timestep_rescaled': t_rescaled.squeeze().item()
        }
        
        return total_loss, info


class VoxelCompleter:
    """
    Main class for SDS-based voxel completion.
    
    This class handles the full optimization loop for completing
    partial voxels using score distillation from TRELLIS.
    """
    
    def __init__(self, device: torch.device = device):
        """
        Initialize the voxel completer.
        
        Args:
            device: torch device for computation
        """
        self.device = device
        self.encoder = None
        self.decoder = None
        self.diffusion = None
        self.pipe = None
        self.is_initialized = False
    
    def initialize_models(self):
        """
        Initialize TRELLIS models for SDS.
        
        This loads:
        - Sparse structure encoder
        - Sparse structure decoder
        - Diffusion model
        """
        import trellis.models as models
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        print("[VoxelCompleter] Loading TRELLIS models...")
        
        # Load encoder
        self.encoder = models.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
        ).to(self.device)
        
        # Load pipeline (includes decoder and diffusion)
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        self.pipe.cuda()
        
        self.decoder = self.pipe.models['sparse_structure_decoder']
        self.diffusion = self.pipe.models['sparse_structure_flow_model']
        
        self.is_initialized = True
        print("[VoxelCompleter] Models initialized")
    
    def prepare_condition(self, image_path: str) -> Dict:
        """
        Prepare conditioning from an input image.
        
        Args:
            image_path: path to conditioning image
            
        Returns:
            conditioning dict for SDS
        """
        if not self.is_initialized:
            self.initialize_models()
        
        image = Image.open(image_path)
        image = self.pipe.preprocess_image(image)
        cond = self.pipe.get_cond([image])
        
        return cond
    
    def complete(self, hashgrid_model: nn.Module,
                 condition: Dict,
                 cfg: SDSConfig = None,
                 output_dir: str = None,
                 known_mask: Optional[torch.Tensor] = None,
                 initial_voxels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, nn.Module]:
        """
        Run SDS optimization to complete voxels with known region preservation.
        
        Args:
            hashgrid_model: HashGridVoxel model representing initial voxels
            condition: conditioning dict from prepare_condition
            cfg: SDS configuration
            output_dir: directory to save intermediate results
            known_mask: (1, 1, 64, 64, 64) binary mask of known regions
            initial_voxels: (1, 1, 64, 64, 64) initial voxel values for preservation
            
        Returns:
            tuple of (completed_voxels, optimized_model)
        """
        if not self.is_initialized:
            self.initialize_models()
        
        cfg = cfg or SDSConfig()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"[VoxelCompleter] Starting SDS optimization for {cfg.total_iters} iterations")
        print(f"[VoxelCompleter] Learning rate: {cfg.lr}")
        print(f"[VoxelCompleter] SDS weight: {cfg.sds_weight}, Voxel weight: {cfg.voxel_weight}, Mask weight: {cfg.mask_weight}")
        
        from network.hash_grid import generate_image_grid_3d, generate_surround_offset
        
        # Setup optimizer with scheduler
        optimizer = torch.optim.Adam(hashgrid_model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.total_iters, eta_min=cfg.lr * 0.1
        )
        
        # Create SDS loss
        sds_loss = SDSLoss(
            encoder=self.encoder,
            decoder=self.decoder,
            diffusion=self.diffusion,
            sampler=self.pipe.sparse_structure_sampler,
            cfg=cfg
        )
        
        # Anti-aliasing setup
        coeff = torch.tensor(cfg.sample_coeff, device=self.device)
        sample_interval = torch.tensor(cfg.sample_interval, device=self.device)
        zero_offset = torch.zeros((1, 1, 3), device=self.device)
        
        # Training history
        history = {'loss': [], 'loss_latent': [], 'loss_voxel': [], 'loss_mask': []}
        
        pbar = tqdm(range(1, cfg.total_iters + 1), desc="SDS Optimization")
        for i in pbar:
            # Generate sampling offset for anti-aliasing
            sample_offset = generate_surround_offset(sample_interval).to(self.device)
            if sample_offset.shape[0] % 2 == 0:
                sample_offset = torch.cat([zero_offset, sample_offset], dim=0)
            
            # Generate grid and predict voxels
            grid = generate_image_grid_3d(64).to(self.device)
            grid = grid.view(-1, 3)
            
            voxels = hashgrid_model(grid).unsqueeze(0)  # (1, 1, 64, 64, 64)
            
            # Compute SDS loss with mask preservation
            loss, info = sds_loss(voxels, condition, known_mask, initial_voxels)
            
            # Record history
            history['loss'].append(loss.item())
            history['loss_latent'].append(info['loss_latent'])
            history['loss_voxel'].append(info['loss_voxel'])
            history['loss_mask'].append(info['loss_mask'])
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % cfg.print_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    latent=f"{info['loss_latent']:.6f}",
                    voxel=f"{info['loss_voxel']:.6f}",
                    mask=f"{info['loss_mask']:.6f}",
                    lr=f"{current_lr:.2e}"
                )
            
            # Save intermediate results
            if output_dir and cfg.save_interval > 0 and i % cfg.save_interval == 0:
                with torch.no_grad():
                    inter_voxels = hashgrid_model(grid).unsqueeze(0)
                    inter_voxels = (inter_voxels > 0.5).float()
                    torch.save(inter_voxels, os.path.join(output_dir, f'voxels_iter{i:06d}.pt'))
        
        # Get final voxels
        with torch.no_grad():
            grid = generate_image_grid_3d(64).to(self.device)
            grid = grid.view(-1, 3)
            final_voxels = hashgrid_model(grid).unsqueeze(0)
            final_voxels = (final_voxels > 0.5).float()
        
        print(f"[VoxelCompleter] Optimization complete")
        print(f"[VoxelCompleter] Final occupancy: {final_voxels.sum().item():.0f} / {64**3}")
        
        # Save training history
        if output_dir:
            import json
            with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f)
        
        return final_voxels, hashgrid_model
    
    def refine_with_diffusion(self, voxels: torch.Tensor,
                              condition: Dict,
                              refine_noise: float = 0.3,
                              num_steps: int = 25) -> torch.Tensor:
        """
        Refine voxels using partial diffusion denoising.
        
        Args:
            voxels: (1, 1, 64, 64, 64) voxel grid
            condition: conditioning dict
            refine_noise: noise level for refinement
            num_steps: number of denoising steps
            
        Returns:
            refined voxel grid
        """
        if not self.is_initialized:
            self.initialize_models()
        
        print(f"[VoxelCompleter] Refining voxels with {num_steps} diffusion steps")
        
        # Encode to latent
        latents = self.encoder(voxels, sample_posterior=False)
        
        # Add noise
        t = torch.ones(1, device=self.device) * refine_noise
        t = 3 * t / (1 + 2 * t)
        t = t[:, None, None, None, None]
        
        noise = torch.randn_like(latents)
        x_t = (1 - t) * latents + t * noise
        
        # Denoise
        refined_latents = self.pipe.sparse_structure_sampler.sample_partial(
            self.diffusion, x_t, refine_noise, num_steps, 3.0,
            **condition,
            cfg_strength=3.0,
            cfg_interval=[0.5, 1.0]
        )
        
        # Decode
        voxels_refined = self.decoder(refined_latents)
        voxels_refined = (voxels_refined > 0.5).float()
        
        return voxels_refined


def run_sds_completion(voxel_data, hashgrid_model: nn.Module,
                       image_path: str,
                       output_dir: str,
                       cfg: SDSConfig = None,
                       preserve_known: bool = True) -> Dict:
    """
    Run the full SDS completion pipeline with known region preservation.
    
    This function:
    1. Generates initial voxels from the hashgrid model
    2. Creates a known mask from the initial voxels
    3. Prepares conditioning from the input image
    4. Runs SDS optimization to complete the voxels
    5. Refines with diffusion denoising
    6. Saves all results
    
    Args:
        voxel_data: GeoSVRVoxelData object (for reference)
        hashgrid_model: HashGridVoxel model initialized with sparse voxels
        image_path: path to conditioning image (e.g., from render_normalize)
        output_dir: output directory
        cfg: SDS configuration
        preserve_known: whether to preserve known regions during optimization
        
    Returns:
        dict with completion results
    """
    from network.hash_grid import generate_image_grid_3d
    
    cfg = cfg or SDSConfig()
    completer = VoxelCompleter()
    
    # Generate initial voxels
    print("[run_sds_completion] Generating initial voxels from hashgrid...")
    grid = generate_image_grid_3d(64).to(device)
    grid = grid.view(-1, 3)
    with torch.no_grad():
        initial_voxels = hashgrid_model(grid).unsqueeze(0)  # (1, 1, 64, 64, 64)
    
    # Create known mask from initial voxels (known = occupied voxels)
    known_mask = None
    initial_voxels_preserved = None
    if preserve_known:
        known_mask = (initial_voxels > 0.5).float()
        initial_voxels_preserved = initial_voxels.detach().clone()
        print(f"[run_sds_completion] Known mask: {known_mask.sum().item():.0f} / {64**3} voxels")
    
    # Prepare conditioning
    print(f"[run_sds_completion] Preparing conditioning from: {image_path}")
    condition = completer.prepare_condition(image_path)
    
    # Run SDS optimization
    print("[run_sds_completion] Starting SDS optimization...")
    completed_voxels, optimized_model = completer.complete(
        hashgrid_model=hashgrid_model,
        condition=condition,
        cfg=cfg,
        output_dir=output_dir,
        known_mask=known_mask,
        initial_voxels=initial_voxels_preserved
    )
    
    # Optionally refine with diffusion
    print("[run_sds_completion] Refining with diffusion...")
    refined_voxels = completer.refine_with_diffusion(
        voxels=completed_voxels,
        condition=condition,
        refine_noise=cfg.refine_noise,
        num_steps=cfg.refine_steps
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    torch.save(optimized_model.state_dict(), 
               os.path.join(output_dir, 'completed_model.pth'))
    torch.save(completed_voxels, 
               os.path.join(output_dir, 'completed_voxels.pt'))
    torch.save(refined_voxels,
               os.path.join(output_dir, 'refined_voxels.pt'))
    
    if preserve_known and known_mask is not None:
        torch.save(known_mask,
                   os.path.join(output_dir, 'known_mask.pt'))
        torch.save(initial_voxels_preserved,
                   os.path.join(output_dir, 'initial_voxels.pt'))
    
    print(f"[run_sds_completion] Results saved to: {output_dir}")
    print(f"[run_sds_completion] Completed voxels: {completed_voxels.sum().item():.0f}")
    print(f"[run_sds_completion] Refined voxels: {refined_voxels.sum().item():.0f}")
    
    return {
        'completed_voxels': completed_voxels,
        'refined_voxels': refined_voxels,
        'optimized_model': optimized_model,
        'known_mask': known_mask,
        'initial_voxels': initial_voxels_preserved
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SDS Completion Module Test")
    print("=" * 60)
    
    # Test config
    cfg = SDSConfig(total_iters=100, print_interval=10)
    print(f"Config: {cfg.__dict__}")
    
    # Note: Full test requires TRELLIS models to be installed
    # and a conditioning image. This is just a structure test.
    print("\nModule loaded successfully. Full test requires TRELLIS installation.")
