#!/usr/bin/env python3
"""
Auto COLMAP reconstruction script for multiple datasets.

Usage:
    # Run on all datasets in root directory
    python run_colmap.py --root_dir /path/to/datasets
    
    # Run on specific dataset
    python run_colmap.py --root_dir /path/to/datasets --dataset_name seq1
    
    # Run with sequential matcher (faster for videos/ordered images)
    python run_colmap.py --root_dir /path/to/datasets --matcher sequential

Error:
    colmap help
    colmap feature_extractor --help   
    colmap exhaustive_matcher --help 
    
"""

import os
import subprocess
import argparse
from pathlib import Path


def run_colmap(basedir, match_type='exhaustive_matcher'):
    """
    Run COLMAP reconstruction on a single dataset.
    
    Args:
        basedir: Path to dataset directory containing 'images' folder
        match_type: 'exhaustive_matcher' or 'sequential_matcher'
    """
    
    # Validate dataset directory
    images_dir = os.path.join(basedir, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist. Skipping {basedir}")
        return False
    
    # Count images
    num_images = len([f for f in os.listdir(images_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if num_images == 0:
        print(f"Error: No images found in {images_dir}. Skipping {basedir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {basedir}")
    print(f"Found {num_images} images")
    print(f"{'='*60}\n")

    # Set environment variables to disable GUI and OpenGL
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['DISPLAY'] = ''
    env['XVFB_WHD'] = '1920x1080x24'
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    try:
        # Step 1: Feature extraction
        print("Step 1/3: Extracting features...")
        feature_extractor_args = [
            'colmap', 'feature_extractor', 
                '--database_path', os.path.join(basedir, 'database.db'), 
                '--image_path', images_dir,
                # '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
                '--FeatureExtraction.use_gpu', '0',
        ]
        feat_output = subprocess.check_output(
            feature_extractor_args, 
            universal_newlines=True, 
            stderr=subprocess.STDOUT,
            env=env
        )
        logfile.write(feat_output)
        print('✓ Features extracted')

        # Step 2: Feature matching
        print(f"Step 2/3: Matching features using {match_type}...")
        matcher_args = [
            'colmap', match_type, 
                '--database_path', os.path.join(basedir, 'database.db'),
                '--FeatureMatching.use_gpu', '0',
        ]
        
        # Add sequential matcher specific options
        if match_type == 'sequential_matcher':
            matcher_args.extend([
                '--SequentialMatching.overlap', '10',
                '--SequentialMatching.loop_detection', '1',
            ])

        match_output = subprocess.check_output(
            matcher_args, 
            universal_newlines=True, 
            stderr=subprocess.STDOUT,
            env=env
        )
        logfile.write(match_output)
        print('✓ Features matched')
        
        # Step 3: Sparse reconstruction
        print("Step 3/3: Building sparse reconstruction...")
        sparse_dir = os.path.join(basedir, 'sparse')
        os.makedirs(sparse_dir, exist_ok=True)

        mapper_args = [
            'colmap', 'mapper',
                '--database_path', os.path.join(basedir, 'database.db'),
                '--image_path', images_dir,
                '--output_path', sparse_dir,
                # '--Mapper.ba_use_gpu', '1',
                # '--Mapper.ba_gpu_index', '1',
        ]
        # No output in command line, because it is redirected to logfile!
        map_output = subprocess.check_output(
            mapper_args, 
            universal_newlines=True, 
            stderr=subprocess.STDOUT,
            env=env
        )
        logfile.write(map_output)
        print('✓ Sparse map created')
        
        model_dir = os.path.join(sparse_dir, '0')
        if os.path.exists(model_dir):
            num_points = 0
            points_file = os.path.join(model_dir, 'points3D.bin')
            if os.path.exists(points_file):
                num_points = os.path.getsize(points_file)
            print(f"✓ Reconstruction successful! Model saved in {model_dir}")
            print(f"  Point cloud size: {num_points / 1024:.2f} KB")
        else:
            print(f"Warning: No reconstruction model found in {model_dir}")
        
        logfile.close()
        print(f'✓ Finished! See {logfile_name} for detailed logs\n')
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running COLMAP on {basedir}:")
        print(e.output)
        logfile.write(f"\nError:\n{e.output}\n")
        logfile.close()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        logfile.close()
        return False


def find_datasets(root_dir):
    """
    Find all valid dataset directories (containing 'images' folder).
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of dataset paths
    """
    datasets = []
    root_path = Path(root_dir)
    
    for item in root_path.iterdir():
        if item.is_dir():
            images_dir = item / 'images'
            if images_dir.exists() and images_dir.is_dir():
                datasets.append(str(item))
    
    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(
        description='Run COLMAP reconstruction on datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all datasets
  python run_colmap.py --root_dir datasets/aea_converted
  
  # Run on specific dataset
  python run_colmap.py --root_dir datasets/aea_converted --dataset_name Seq1
  
  # Use sequential matcher (faster for video sequences)
  python run_colmap.py --root_dir datasets/aea_converted --matcher sequential
        """
    )
    
    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help='Root directory containing dataset folders'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Specific dataset name to process (default: process all)'
    )
    
    parser.add_argument(
        '--matcher',
        type=str,
        default='exhaustive',
        choices=['exhaustive', 'sequential'],
        help='Matcher type: exhaustive (accurate) or sequential (fast for videos)'
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory {args.root_dir} does not exist")
        return
    
    # Determine matcher command
    matcher_cmd = 'exhaustive_matcher' if args.matcher == 'exhaustive' else 'sequential_matcher'
    
    # Find datasets
    if args.dataset_name:
        # Process specific dataset
        dataset_path = os.path.join(args.root_dir, args.dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset {dataset_path} does not exist")
            return
        datasets = [dataset_path]
    else:
        # Process all datasets
        datasets = find_datasets(args.root_dir)
        if not datasets:
            print(f"No valid datasets found in {args.root_dir}")
            print("(Datasets should contain 'images' folder)")
            return
    
    print(f"\nFound {len(datasets)} dataset(s) to process:")
    for ds in datasets:
        print(f"  - {os.path.basename(ds)}")
    print()
    
    # Process each dataset
    success_count = 0
    failed_datasets = []
    
    for dataset in datasets:
        success = run_colmap(dataset, matcher_cmd)
        if success:
            success_count += 1
        else:
            failed_datasets.append(os.path.basename(dataset))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total datasets: {len(datasets)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        print("\nFailed datasets:")
        for name in failed_datasets:
            print(f"  - {name}")
    
    print()


if __name__ == '__main__':
    main()
