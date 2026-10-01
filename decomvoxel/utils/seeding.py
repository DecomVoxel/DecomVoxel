import random
import numpy as np
import torch
import hashlib 

DEFAULT_SEED = 0

def seed_everything(seed=DEFAULT_SEED):
    """Set random seeds for Python, NumPy, PyTorch, and sklearn based on a string or int."""
    # Convert string to an integer hash
    if isinstance(seed, str):
        seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32) # 32-bit seed
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def seed_blender(seed=DEFAULT_SEED, denoising=False):
    
    import bpy
    random.seed(seed)
    np.random.seed(seed) 
     
    bpy.context.scene.cycles.seed = seed             
    bpy.context.scene.cycles.use_animated_seed = False     
    bpy.context.scene.cycles.use_adaptive_sampling = False 
    bpy.context.scene.cycles.sampling_pattern = 'SOBOL_BURLEY'  
    
    if denoising:
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.scene.cycles.use_denoising = True
    else:
        bpy.context.view_layer.cycles.use_denoising = False
        bpy.context.scene.cycles.use_denoising = False