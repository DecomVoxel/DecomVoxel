#!/bin/bash
# COLMAP reconstruction script for datasets with automatic image resizing
# Usage: bash dataset_tools/run_colmap.bash <root_dir> [dataset_name] [target_width] [target_height] [downsample]
# 
# Examples:
#   bash dataset_tools/run_colmap.bash datasets/on-the-go mountain
#   bash dataset_tools/run_colmap.bash datasets/on-the-go mountain 1600 1200
#   bash dataset_tools/run_colmap.bash datasets/on-the-go mountain 0 0 2  
#   bash dataset_tools/run_colmap.bash datasets/aea_sample


ROOT_DIR=$1
DATASET_NAME=$2
TARGET_WIDTH=${3:-800}
TARGET_HEIGHT=${4:-600}
DOWNSAMPLE=${5:-1}

if [ -z "$ROOT_DIR" ]; then
    echo "Usage: bash $0 <root_dir> [dataset_name] [target_width] [target_height] [downsample]"
    echo "  If downsample > 1, target_width and target_height will be auto-calculated from original image size"
    exit 1
fi

# Auto-calculate target resolution if downsample is specified
if [ "$DOWNSAMPLE" -gt 1 ]; then
    echo "Downsample factor: $DOWNSAMPLE"
    
    # Find a sample image to get original dimensions
    if [ -n "$DATASET_NAME" ]; then
        SAMPLE_DIR="$ROOT_DIR/$DATASET_NAME/images"
    else
        SAMPLE_DIR=$(find "$ROOT_DIR" -type d -name "images" | head -1)
    fi
    
    if [ -d "$SAMPLE_DIR" ]; then
        SAMPLE_IMAGE=$(find "$SAMPLE_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.JPG" -o -name "*.PNG" \) | head -1)
        
        if [ -n "$SAMPLE_IMAGE" ]; then
            echo "Sample image: $SAMPLE_IMAGE"
            # Get original image dimensions using identify (ImageMagick) or Python
            if command -v identify &> /dev/null; then
                ORIG_WIDTH=$(identify -format "%w" "$SAMPLE_IMAGE")
                ORIG_HEIGHT=$(identify -format "%h" "$SAMPLE_IMAGE")
            else
                # Fallback to Python
                DIMS=$(python3 -c "from PIL import Image; img = Image.open('$SAMPLE_IMAGE'); print(img.width, img.height)")
                ORIG_WIDTH=$(echo $DIMS | cut -d' ' -f1)
                ORIG_HEIGHT=$(echo $DIMS | cut -d' ' -f2)
            fi
            
            if [ -n "$ORIG_WIDTH" ] && [ -n "$ORIG_HEIGHT" ] && [ "$ORIG_WIDTH" -gt 0 ] && [ "$ORIG_HEIGHT" -gt 0 ]; then
                TARGET_WIDTH=$((ORIG_WIDTH / DOWNSAMPLE))
                TARGET_HEIGHT=$((ORIG_HEIGHT / DOWNSAMPLE))
                
                echo "Original resolution: ${ORIG_WIDTH}x${ORIG_HEIGHT}"
                echo "Target resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
            else
                echo "Error: Could not determine image dimensions"
                exit 1
            fi
        else
            echo "Error: No image files found in $SAMPLE_DIR"
            exit 1
        fi
    else
        echo "Error: Images directory not found: $SAMPLE_DIR"
        exit 1
    fi
fi

# Verify target dimensions are valid
if [ "$TARGET_WIDTH" -le 0 ] || [ "$TARGET_HEIGHT" -le 0 ]; then
    echo "Error: Invalid target resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
    exit 1
fi

echo "============================================================"
echo "Step 0: Resizing images to ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "============================================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run resize script
if [ -n "$DATASET_NAME" ]; then
    python3 "$SCRIPT_DIR/utils/resize_images.py" "$ROOT_DIR" "$DATASET_NAME" --width "$TARGET_WIDTH" --height "$TARGET_HEIGHT"
else
    python3 "$SCRIPT_DIR/utils/resize_images.py" "$ROOT_DIR" --width "$TARGET_WIDTH" --height "$TARGET_HEIGHT"
fi

echo ""
echo "============================================================"
echo "Starting COLMAP reconstruction"
echo "============================================================"
echo ""

# Set environment variables
export QT_QPA_PLATFORM=offscreen
export DISPLAY=
export XVFB_WHD=1920x1080x24

# Get list of datasets
if [ -n "$DATASET_NAME" ]; then
    # Single dataset
    DATASETS=("$ROOT_DIR/$DATASET_NAME")
else
    # All datasets with images folder
    DATASETS=()
    for dir in "$ROOT_DIR"/*; do
        if [ -d "$dir/images" ]; then
            DATASETS+=("$dir")
        fi
    done
fi

echo "Found ${#DATASETS[@]} dataset(s) to process"
echo ""

# Process each dataset
for BASEDIR in "${DATASETS[@]}"; do
    DATASET=$(basename "$BASEDIR")
    IMAGES_DIR="$BASEDIR/images"
    DATABASE_PATH="$BASEDIR/database.db"
    SPARSE_DIR="$BASEDIR/sparse"
    LOG_FILE="$BASEDIR/colmap_output.txt"
    
    echo "============================================================"
    echo "Processing: $DATASET"
    echo "============================================================"
    
    # Skip if no images folder
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "✗ No images folder found. Skipping."
        continue
    fi
    
    # Clean and create directories
    rm -f "$DATABASE_PATH"
    rm -rf "$SPARSE_DIR"
    mkdir -p "$SPARSE_DIR"
    
    # Step 1: Feature extraction
    echo "Step 1/3: Extracting features..."
    colmap feature_extractor \
        --database_path "$DATABASE_PATH" \
        --image_path "$IMAGES_DIR" \
        --ImageReader.camera_model SIMPLE_PINHOLE \
        --FeatureExtraction.use_gpu 1 \
        > "$LOG_FILE" 2>&1
    
    echo "✓ Done"
    
    # Step 2: Feature matching
    echo "Step 2/3: Matching features..."
    colmap exhaustive_matcher \
        --database_path "$DATABASE_PATH" \
        --FeatureMatching.use_gpu 1 \
        >> "$LOG_FILE" 2>&1
    
    echo "✓ Done"
    
    # Step 3: Sparse reconstruction
    echo "Step 3/3: Building reconstruction..."
    colmap mapper \
        --database_path "$DATABASE_PATH" \
        --image_path "$IMAGES_DIR" \
        --output_path "$SPARSE_DIR" \
        >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed. See $LOG_FILE"
        continue
    fi
    
    # Check result
    if [ -d "$SPARSE_DIR/0" ]; then
        echo "✓ Reconstruction complete! Saved in $SPARSE_DIR/0"
    else
        echo "✗ No reconstruction found"
    fi
    
    echo ""
done

echo "============================================================"
echo "All datasets processed!"
echo "============================================================"
