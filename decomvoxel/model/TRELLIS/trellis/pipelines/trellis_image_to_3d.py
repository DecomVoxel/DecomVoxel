from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult
from contextlib import contextmanager

# Extra
import math
import imageio
import matplotlib.pyplot as plt # type: ignore
import kaolin as kal
import lpips
from trellis.utils.render_utils import render_multiview_opt

def rotate_vertices(vertices, axis='y', angle_degrees=45.0):
    """
    Rotate vertices around a given axis (x, y, or z) by a specified angle in degrees.
    """
    angle = math.radians(angle_degrees)
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)

    # Rotation matrices for each axis
    if axis == 'x':
        R = torch.tensor([
            [1,       0,        0],
            [0,  cos_val, -sin_val],
            [0,  sin_val,  cos_val]
        ], dtype=vertices.dtype, device=vertices.device)
    elif axis == 'y':
        R = torch.tensor([
            [ cos_val, 0, sin_val],
            [       0, 1,       0],
            [-sin_val, 0, cos_val]
        ], dtype=vertices.dtype, device=vertices.device)
    elif axis == 'z':
        R = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val,  cos_val, 0],
            [      0,        0, 1]
        ], dtype=vertices.dtype, device=vertices.device)
    else:
        raise ValueError("Axis must be one of 'x', 'y', 'z'.")

    # Apply rotation: [N, 3] = [N, 3] * [3, 3]
    rotated_vertices = vertices @ R.T
    return rotated_vertices

def visualize_voxels(p, z, size=64, prefix='noise'):

    # z = (z - z.min()) / (z.max() - z.min())
    # Convert torch Tensors to numpy arrays
    
    p_np = p.cpu().numpy().astype(int)
    z_np = z.cpu().numpy()

    # Initialize a 3D boolean array to mark which positions contain voxels
    cube = np.zeros((size, size, size), dtype=bool)

    # Initialize a 4-channel color array (RGBA)
    facecolors = np.zeros((size, size, size, 4), dtype=np.float32)

    # Populate the arrays
    for i in range(p_np.shape[0]):
        x, y, z_ = p_np[i]
        cube[x, y, z_] = True
        r, g, b = z_np[i]
        facecolors[x, y, z_] = (r, g, b, 1.0)  # Full opacity

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels
    ax.voxels(cube, facecolors=facecolors, edgecolor='k')

    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)

    plt.tight_layout()
    plt.savefig(f'debug/backpack/voxels_{prefix}.png')
    
def visualize_dense_voxels(data, size=16, prefix='noise_low'):

    # Convert torch Tensor to numpy if needed
    if not isinstance(data, np.ndarray):
        data = data.cpu().numpy()
    data = (data - data.min()) / (data.max() - data.min())
    
    # data is [64,64,64,3]
    size = data.shape[0]

    # Determine which voxels are present. For visualization, let's consider any voxel 
    # with a nonzero color as present.
    cube = np.any(data > 0, axis=-1)  # shape: [64,64,64]

    # Create a facecolors array with an alpha channel
    facecolors = np.zeros((size, size, size, 4), dtype=data.dtype)
    facecolors[..., :3] = data
    facecolors[..., 3] = 1.0  # full opacity

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(cube, facecolors=facecolors, edgecolor='k')

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)

    plt.tight_layout()
    plt.savefig(f'debug/backpack/voxels_{prefix}.png')

def lpips_loss(pred, target, lpips_fun):
    return lpips_fun(pred, target).flatten()

class LPIPSLoss(nn.Module):

    def __init__(self,
                 net='vgg',
                 lpips_list=None,
                 normalize_inputs=True,
                 loss_weight=1.0):
        super().__init__()
        self.net = net
        self.lpips = [] if (lpips_list is None or lpips_list[0].pnet_type != net) else lpips_list  # use a list to avoid registering the LPIPS model in state_dict
        self.normalize_inputs = normalize_inputs
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        dtype = pred.dtype
        cdtype = torch.bfloat16
        if len(self.lpips) == 0:
            lpips_eval = lpips.LPIPS(
                net=self.net, eval_mode=True, pnet_tune=False).to(
                device=pred.device, dtype=cdtype)
            # with torch.no_grad():
            #     lpips_eval = torch.jit.trace(lpips_eval, (pred.to(cdtype), target.to(cdtype)))
                # lpips_eval = torch.jit.optimize_for_inference(lpips_eval)
            self.lpips.append(lpips_eval)
        if self.normalize_inputs:
            pred = pred * 2 - 1
            target = target * 2 - 1
        with torch.jit.optimized_execution(False):
            return lpips_loss(
                pred.to(cdtype), target.to(cdtype), lpips_fun=self.lpips[0]
            ).to(dtype) * self.loss_weight
    
class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        
    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        # output = output.crop(bbox)
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        # output[:20] = 1
        # output[-20:] = 1
        # output[:, :20] = 1
        # output[:, -20:] = 1
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        decoded = decoder(z_s) 
        coords = torch.argwhere(decoded > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        noise=None,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        
        if noise is None:
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
                )
        
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        slat_samples = slat.samples
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat_samples.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat_samples.device)
        slat_samples = slat_samples * std + mean
        
        return slat_samples

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        coords=None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        # mesh = kal.io.obj.import_mesh('dataset/backpack/backpack.obj',
        #     with_normals=False, with_materials=False, error_handler=None)
        # vertices = mesh.vertices
        # vertices = rotate_vertices(vertices, axis='z', angle_degrees=-60.0)
        # faces = mesh.faces
        # vertices = vertices.unsqueeze(0)  # Shape: [1, N, 3]
        # voxel_grid = kal.ops.conversions.trianglemeshes_to_voxelgrids(
        #     vertices, faces, resolution=64, return_sparse=False).unsqueeze(0)
        # coords = torch.argwhere(voxel_grid > 0)[:, [0, 2, 3, 4]].int().to(self.device)
        # print(coords.shape)
        # visualize_voxels(coords[:, 1:], torch.ones_like(coords)[:, :3] * 0.5, 64, 'occ2')
        
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        
        if coords is None:
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)

        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)  
    
    def optimization(
        self,
        task_name: str,
        test_name: str,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        
        opt_iters = 10001
        num_views = 30
        
        masks = []
        gts = []
        
        for i in range(num_views):
            mask = Image.open(f'debug/{test_name}/data/color_{i}.png')
            mask = torch.from_numpy(np.array(mask)[:, :, 3:]).permute(2, 0, 1).float().unsqueeze(0) / 255
            mask = mask.to(self.device)
            gt = Image.open(f'debug/{test_name}/data/refined_{i}.png')
            gt = torch.from_numpy(np.array(gt)[:, :, :3]).permute(2, 0, 1).float().unsqueeze(0) / 255
            gt = gt.to(self.device)
            masks.append(mask)
            gts.append(gt)
            
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        # cond['cond'] = torch.zeros_like(cond['cond'])
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        
        slat.feats.requires_grad = True
        optimizer = torch.optim.Adam([slat.feats], lr=5e-3, betas=[0.9, 0.99], weight_decay=0)
        loss_lpips = LPIPSLoss(loss_weight=1.2, net='vgg')
        loss_pixel = nn.L1Loss()
        # loss_pixel = nn.MSELoss()
        
        for i in range(opt_iters):
            
            view_idx = np.random.randint(num_views)
            gs = self.decode_slat(slat, ['gaussian'])
            color = render_multiview_opt(gs['gaussian'][0], view_idx=view_idx, resolution=1024, nviews=30).unsqueeze(0)
            loss = loss_lpips(color, gts[view_idx]) + loss_pixel(color * masks[view_idx], gts[view_idx] * masks[view_idx])
            # loss = loss_pixel(color * mask, gt * mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"[{i:04d}] Loss: {loss.item()} Feat: {slat.feats.mean().item()}")
            
            if i % 500 == 0:
                color = torch.cat([color, masks[view_idx]], dim=1)
                color = color[0].detach().cpu().numpy().transpose(1, 2, 0)
                color = (color.clip(0, 1) * 255).astype(np.uint8)
                imageio.imwrite(f'debug/{test_name}/{task_name}/optimized_{i}.png', color)
            
        return self.decode_slat(slat, formats)    

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            # cond_indices = (np.arange(num_steps) % num_images).tolist()
            cond_indices = ((np.arange(num_steps) + 7) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)  
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return coords, self.decode_slat(slat, formats)
        