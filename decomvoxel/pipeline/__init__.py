from decomvoxel.pipeline.load_voxel import (
    # Voxel data containers
    GeoSVRVoxelData,
    SparseStructure,
    # Loaders
    VoxelLoader,
    # Converters
    convert_voxel_to_sparse_structure,
    convert_voxel_batch,
    # VoxelToHashGridConverter,
    # ObjectVoxelProcessor,
    # Convenience functions
    load_object_voxel,
    voxel_to_sparse_structure,
    # load_sparse_structure,
    # load_and_convert_to_sparse_structure,
    # convert_to_hashgrid
)

from decomvoxel.pipeline.score_distillation_sampling_ss import (
    SDSConfig,
    SparseStructureCompleter,
    run_sds_completion_ss
)

# Legacy imports for backward compatibility
from decomvoxel.pipeline.score_distillation_sampling_hg import (
    SDSConfig as LegacySDSConfig,
    SDSLoss,
    VoxelCompleter,
    run_sds_completion
)

from decomvoxel.pipeline.visualize_sparse_structure import (
    SparseStructureVisualizer,
    visualize_sparse_structure,
    compare_sparse_structures
)

from decomvoxel.pipeline.visualize_hash_grid import (
    HashGridVisualizer,
    HashGridToMeshConverter,
    visualize_hashgrid,
    hashgrid_to_mesh
)

from decomvoxel.pipeline.sparse_structure_to_mesh import (
    SSToMeshConfig,
    SparseStructureToMeshPipeline,
    sparse_structure_to_mesh,
    coords_to_mesh,
)

__all__ = [
    # Voxel data containers
    'GeoSVRVoxelData',
    'SparseStructure',
    # Loaders
    'VoxelLoader',
    # Converters
    'convert_voxel_to_sparse_structure',
    'convert_voxel_batch',
    'VoxelToHashGridConverter',
    'ObjectVoxelProcessor',
    # Convenience functions
    'load_object_voxel',
    'voxel_to_sparse_structure',
    'load_sparse_structure',
    'load_and_convert_to_sparse_structure',
    'convert_to_hashgrid',
    # SDS completion (new sparse structure based)
    'SDSConfig',
    'SparseStructureSDS',
    'SparseStructureCompleter',
    'run_sds_completion_ss',
    # SDS completion (legacy HashGrid based)
    'LegacySDSConfig',
    'SDSLoss',
    'VoxelCompleter',
    'run_sds_completion',
    # Sparse structure visualization
    'SparseStructureVisualizer',
    'visualize_sparse_structure',
    'compare_sparse_structures',
    # HashGrid visualization
    'HashGridVisualizer',
    'HashGridToMeshConverter',
    'visualize_hashgrid',
    'hashgrid_to_mesh',
    # Sparse structure to mesh
    'SSToMeshConfig',
    'SparseStructureToMeshPipeline',
    'sparse_structure_to_mesh',
    'coords_to_mesh',
]
