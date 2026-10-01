import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GridTransformer():

    def __init__(self, axis_type, axis_dir, opt_dir=False):

        self.axis_type = axis_type
        if opt_dir:
            self.axis_dir = axis_dir[None, None, :]
        else:
            self.axis_dir = torch.tensor(axis_dir).to(device).float()[None, None, :]

    def translate_grid(self, grid, qpos):
        grid = grid + qpos * self.axis_dir 
        return grid

    def rotate_grid(self, grid, qpos, axis_point):

        if grid.shape[0] == 1:
            grid = grid[0]
        qpos = qpos[0, 0, 0]
        axis_dir = self.axis_dir[0, 0]
        
        # Normlization
        axis_dir = axis_dir / torch.norm(axis_dir, p=2)
        axis_point = axis_point / 63. - 0.5
        
        # Translate grid to the origin relative to axis_point
        grid_shifted = grid - axis_point

        # Rodrigues' rotation formula components
        cos_theta = torch.cos(qpos)
        sin_theta = torch.sin(qpos)
        one_minus_cos = 1 - cos_theta

        ux, uy, uz = axis_dir

        # Rotation matrix using Rodrigues' formula
        R = torch.stack([
            torch.stack([cos_theta + ux**2 * one_minus_cos, ux * uy * one_minus_cos - uz * sin_theta, ux * uz * one_minus_cos + uy * sin_theta]),
            torch.stack([uy * ux * one_minus_cos + uz * sin_theta, cos_theta + uy**2 * one_minus_cos, uy * uz * one_minus_cos - ux * sin_theta]),
            torch.stack([uz * ux * one_minus_cos - uy * sin_theta, uz * uy * one_minus_cos + ux * sin_theta, cos_theta + uz**2 * one_minus_cos])
        ])  
        
        # Apply rotation
        rotated_shifted = torch.matmul(grid_shifted, R.T)

        # Translate back
        rotated_grid = rotated_shifted + axis_point

        return rotated_grid[None, :, :]

    def transform(self, grid, qpos, axis_point=None, qpos_scale=1):
         
        qpos = qpos[:, None, None] * qpos_scale
        if self.axis_type == 0:
            grid = self.translate_grid(grid, qpos)
        elif self.axis_type == 1:
            grid = self.rotate_grid(grid, qpos, axis_point)
        return grid

def remove_isolated_voxels(occupancy):
    """
    Sets occupancy[i,j,k] = 0 if all its 6-connected neighbors are 0.
    
    Args:
        occupancy (torch.Tensor): Input occupancy grid of shape (D, H, W)
    Returns:
        torch.Tensor: Processed occupancy grid
    """
    assert occupancy.shape == (64, 64, 64), "Expected shape (64, 64, 64)"

    # Ensure it's float for convolution
    occ = occupancy.float().unsqueeze(0).unsqueeze(0)  # shape (1, 1, D, H, W)

    # Define 6-neighbour convolution kernel
    kernel = torch.zeros((1, 1, 3, 3, 3), device=occ.device)
    kernel[0, 0, 1, 1, 0] = 1  # left
    kernel[0, 0, 1, 1, 2] = 1  # right
    kernel[0, 0, 1, 0, 1] = 1  # up
    kernel[0, 0, 1, 2, 1] = 1  # down
    kernel[0, 0, 0, 1, 1] = 1  # front
    kernel[0, 0, 2, 1, 1] = 1  # back

    # Apply convolution to count nonzero neighbors
    neighbor_count = F.conv3d(occ, kernel, padding=1)

    # Keep only voxels with at least one occupied neighbor
    keep_mask = (neighbor_count > 0).squeeze(0).squeeze(0)
    result = occupancy * keep_mask

    return result

def calc_grid_weight(grid_sample, grid, coeff=2):
    
    dist = torch.norm(grid_sample - grid, dim=-1) * 63
    weight = 1.0 / ((1.0 + dist * coeff) ** 2)  
    weight = weight / torch.sum(weight, dim=0, keepdim=True) 
    return weight.reshape(weight.shape[0], -1, 64, 64, 64)
    
    
    
        