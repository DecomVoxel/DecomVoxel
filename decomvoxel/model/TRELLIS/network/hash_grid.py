import torch
import sys
sys.path.append('./')

from torch import nn
from torch.nn import functional as F
from network.tcnn import get_encoding, get_mlp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_image_grid(height, width, normalized=True):
    xs = torch.linspace(0, 1, steps=width) if normalized else torch.arange(width)
    ys = torch.linspace(0, 1, steps=height) if normalized else torch.arange(height)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((grid_x, grid_y), dim=-1)

def generate_surround_offset(sample_interval=0.5, step=3):
    xs = torch.linspace(-sample_interval, sample_interval, steps=step)
    ys = torch.linspace(-sample_interval, sample_interval, steps=step)
    zs = torch.linspace(-sample_interval, sample_interval, steps=step)
    offset_z, offset_y, offset_x = torch.meshgrid(zs, ys, xs, indexing='ij')
    offset = torch.stack((offset_x, offset_y, offset_z), dim=-1)
    offset = offset.view(-1, 1, 1, 3) / 63.
    return offset

# The dense tensor uses the spatial layout [Z, Y, X].
def generate_image_grid_3d(res, normalized=True):
    xs = torch.linspace(-0.5, 0.5, steps=res)
    ys = torch.linspace(-0.5, 0.5, steps=res)
    zs = torch.linspace(-0.5, 0.5, steps=res) 
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing='ij')
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)

def scale_tensor(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def contract_to_unisphere(x, bbox, unbounded=False):
    return scale_tensor(x, bbox, (0, 1))

class HashGrid(nn.Module):

    def __init__(self):
        super().__init__()

        n_feature_dims = 3
        pos_encoding_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
            # default_factory=lambda: {
            #     "otype": "Frequency", 
	        #     "n_frequencies": 10
            # }
        
        mlp_network_config: dict = {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        self.encoding = get_encoding(2, pos_encoding_config)
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            n_feature_dims,
            mlp_network_config,
        )
        
        
    def forward(self, points):
        enc = self.encoding(points.view(-1, 2))
        rgbs = self.feature_network(enc).view(*points.shape[:-1], 3)
        return rgbs

class HashGridVoxel(nn.Module):

    def __init__(self):
        super().__init__()

        n_feature_dims = 1
        pos_encoding_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        
        mlp_network_config: dict = {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "sigmoid",
                # "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        
        self.encoding = get_encoding(3, pos_encoding_config)
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            n_feature_dims,
            mlp_network_config,
        )
    
    def forward(self, points):
        enc = self.encoding(points)
        voxels = self.feature_network(enc).view(-1, 64, 64, 64)
        return voxels

if __name__ == '__main__':
    
    res = 64
    network = HashGridVoxel().to(device)
        
    grid = generate_image_grid_3d(res)
    grid = grid.to(device)
    with torch.enable_grad():
        voxels = network(grid, rot=True)
        loss = voxels.sum()  # Use sum() to make sure gradient is not zero
        loss.backward()
        print("rot_z.grad:", network.rot_z.grad)  # Should not be None or zero
