import torch
import numpy as np

def flood_fill_exterior(mask):
    """
    Flood fill from the boundaries to identify exterior regions.
    
    Args:
        mask: Boolean tensor with True indicating occupied cells
        
    Returns:
        Boolean tensor with True indicating exterior cells
    """
    # Create a copy of the mask padded with an exterior boundary
    h, w = mask.shape
    padded = torch.zeros((h + 2, w + 2), dtype=torch.bool, device=mask.device)
    padded[1:h+1, 1:w+1] = mask
    
    # Initialize queue with boundary points
    exterior = torch.zeros_like(padded)
    queue = []
    
    # Add all boundary points to the queue
    for i in range(padded.shape[0]):
        queue.append((i, 0))
        queue.append((i, padded.shape[1] - 1))
    
    for j in range(padded.shape[1]):
        queue.append((0, j))
        queue.append((padded.shape[0] - 1, j))
    
    # Perform flood fill
    while queue:
        i, j = queue.pop(0)
        
        if 0 <= i < padded.shape[0] and 0 <= j < padded.shape[1] and not padded[i, j] and not exterior[i, j]:
            exterior[i, j] = True
            
            # Add neighbors to the queue
            queue.append((i + 1, j))
            queue.append((i - 1, j))
            queue.append((i, j + 1))
            queue.append((i, j - 1))
    
    # Return the unpadded exterior mask
    return exterior[1:h+1, 1:w+1]

def remove_disk(voxels):

    first_del = True
    coords = np.argwhere(voxels > 0)[:, [0, 2, 3, 4]]
    disk_cond = np.logical_or(coords[:, 1] == 0, coords[:, 2] == 0)

    while np.sum(disk_cond) > 0:

        coords_disk = coords[disk_cond]
        disk_height = np.min(coords_disk[:, 3]).astype(int)
        
        if np.sum(voxels[:, :, :, :, disk_height]) < 100:
            voxels[:, :, :, :, disk_height] = 0
            coords = np.argwhere(voxels > 0)[:, [0, 2, 3, 4]]
            disk_cond = np.logical_or(coords[:, 1] == 0, coords[:, 2] == 0)
            continue
        
        if first_del:
            voxels[:, :, :, :, :disk_height+1] = 0 
        else:
            voxels[:, :, :, :, disk_height] = 0 
        
        first_del = False
        coords = np.argwhere(voxels > 0)[:, [0, 2, 3, 4]]
        disk_cond = np.logical_or(coords[:, 1] == 0, coords[:, 2] == 0)
    
    return voxels

def remove_below_disk(voxels):

    coords = torch.argwhere(voxels > 0)[:, [0, 2, 3, 4]].int()
    disk_cond = torch.logical_or(coords[:, 1] == 0, coords[:, 2] == 0)

    if disk_cond.sum() > 0:

        coords_disk = coords[disk_cond]
        disk_height = coords_disk[:, 3].min().item()
        voxels[:, :, :, :, :disk_height] = 0 

    return voxels